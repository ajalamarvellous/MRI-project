import logging
import os
import time
from typing import Callable

import torch
import tqdm
from livelossplot import PlotLosses
from sklearn.metrics import accuracy_score, precision_score

import mlflow

logging.basicConfig(
    level=logging.INFO,
    # fmt: off
    format="%(asctime)s %(funcName)s[%(levelname)s]: %(message)s",
    # fmt: on
)
logger = logging.getLogger()


class Trainer:
    def __init__(
        self,
        epochs: int,
        lr: float,
        model: Callable,
        criterion: Callable,
        optimiser: Callable,
        device: torch.device,
        model_dir: str,
        result_dir: str,
        metrics: dict = {},
        track_mlflow: bool = True,
    ):
        self.model = model
        self.criterion = criterion
        self.optimiser = optimiser
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.metrics = metrics
        self.plotlosses = PlotLosses()
        self.track_mlflow = track_mlflow
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        self.model_dir = model_dir
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        self.result_dir = result_dir

    def compute_accuracy(self, model, data_loader):
        # correct_pred, num_examples = 0, 0
        loss, counter = 0, 0
        y_true, y_logit = [], []
        for features, targets in tqdm.tqdm(data_loader, desc="Evaluating..."):
            features = features.to(self.device)
            targets = targets.view(-1, 1).to(self.device)

            probas, predicted_labels = self.predict(model, features)
            cost = self.criterion(probas, targets.type(torch.float))

            y_true.extend(list(targets.numpy()))
            y_logit.extend(torch.where(predicted_labels > 0.5, 1, 0).detach().tolist())
            # num_examples += targets.size(0)
            # correct_pred += (predicted_labels == targets).sum()
            counter += 1
            loss += cost.detach().numpy()
        # accuracy = correct_pred.float() / num_examples * 100
        accuracy = accuracy_score(y_true, y_logit)
        precision = precision_score(y_true, y_logit)
        loss /= counter
        metrics = {"Val_loss": loss, "Accuracy": accuracy, "Precision": precision}
        return metrics

    def predict(self, model, data):
        probas = model(data)
        predicted_labels = torch.where(probas > 0.5, 1, 0)
        return probas, predicted_labels

    def train(self, train_loader, test_loader, eval_every=5):
        logger.info("Starting training now...")
        for epoch in range(self.epochs):
            self.model = self.model.train()
            loss = 0
            train_counter = 0
            start_time = time.time()
            for features, targets in tqdm.tqdm(train_loader, desc="Training"):
                train_counter += 1
                # move data to device
                features = features.to(self.device)
                targets = targets.view(-1, 1).to(self.device)
                ### FORWARD AND BACK PROP
                probas = self.model(features)
                cost = self.criterion(probas, targets.type(torch.float))
                loss += cost.detach().numpy()
                self.optimiser.zero_grad()
                cost.backward()
                ### UPDATE MODEL PARAMETERS
                self.optimiser.step()
            loss = loss / train_counter
            self.model = self.model.eval()
            time_per_epoch = (time.time() - start_time) / 60
            if (epoch % eval_every == 0) or epoch == self.epochs - 1:
                self.metrics = self.compute_accuracy(self.model, test_loader)
                self.save_model(epoch)
            self.metrics.update({"Train_loss": loss})
            self.metrics.update({"Epoch": epoch})
            self.metrics.update({"Time_per_epoch": time_per_epoch})
            logger.info(
                f"Epoch: {epoch + 1}/{self.epochs} \n accuracy: \
                        {self.metrics['Accuracy']} \n precision: \
                        {self.metrics['Precision']} \n Training loss: \
                        {self.metrics['Train_loss']} \n Validation loss: \
                        {self.metrics['Val_loss']} \n Time elapsed: \
                        {time_per_epoch}"
            )
            self.plotlosses.update({"Train_loss": loss})
            self.plotlosses.update({"val_loss": self.metrics["Val_loss"]})
            if self.track_mlflow:
                self.mlflow_tracking(epoch)
        logger.info("Training completed now ...")
        # self.plotlosses.send()

    def mlflow_tracking(self, epoch):
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_name", self.model._get_name())
            mlflow.log_param("lr", self.lr)
            mlflow.log_param("epochs", epoch)

            mlflow.log_metric("Accuracy", self.metrics["Accuracy"])
            mlflow.log_metric("Training_loss", self.metrics["Train_loss"])
            mlflow.log_metric("Validation_loss", self.metrics["Val_loss"])

    def save_model(self, epoch):
        dir_ = self.model_dir + f"/{self.model._get_name()}_{epoch}.pth"
        torch.save(
            {
                "model_checkpoint": self.model.state_dict(),
                "optimiser_checkpoint": self.optimiser.state_dict(),
            },
            dir_,
        )
        logger.info(f"Model saved to {dir_}")
