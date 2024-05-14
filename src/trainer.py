import logging
import time
from typing import Callable

import torch
import tqdm
from livelossplot import PlotLosses

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
        metrics: dict = {},
    ):
        self.model = model
        self.criterion = criterion
        self.optimiser = optimiser
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.metrics = metrics
        self.plotlosses = PlotLosses()

    def compute_accuracy(self, model, data_loader):
        correct_pred, num_examples = 0, 0
        loss, counter = 0, 0
        for features, targets in tqdm.tqdm(data_loader, desc="Evaluating..."):
            features = features.to(self.device)
            targets = targets.view(-1, 1).to(self.device)

            probas, predicted_labels = self.predict(model, features)
            cost = self.criterion(probas, targets.type(torch.float))

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
            counter += 1
            loss += cost.detach().numpy()
        accuracy = correct_pred.float() / num_examples * 100
        loss /= counter
        metrics = {"Val_loss": loss, "Accuracy": accuracy}
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
            if epoch % eval_every == 0:
                self.metrics = self.compute_accuracy(self.model, test_loader)
            self.metrics.update({"Train_loss": loss})
            self.metrics.update({"Epoch": epoch})
            self.metrics.update({"Time_per_epoch": time_per_epoch})
            logger.info(
                f"Epoch: {epoch + 1}/{self.epochs} accuracy: \
                        {self.metrics['Accuracy']} \n Training loss: \
                        {self.metrics['Train_loss']} \n Validation loss: \
                        {self.metrics['Val_loss']} \n Time elapsed: \
                        {time_per_epoch}"
            )
            self.plotlosses.update({"Train_loss": loss})
            self.plotlosses.update({"val_loss": self.metrics["Val_loss"]})
            self.plotlosses.send()
        logger.info("Training completed now ...")
