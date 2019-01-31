import torch
from model import FC4
from load import run_loader, calculate_data_size
from tqdm import tqdm
from sklearn.metrics import classification_report


class Testing:
    def __init__(self, trainedModel, pathTest, batchSize, imageSize):
        super(Testing, self).__init__()
        self.model = FC4()
        state_dict = torch.load(trainedModel)
        self.model.load_state_dict(state_dict)
        self.pathTest = pathTest
        self.imageSize = imageSize
        self.batchSize = batchSize
        self.data_size = calculate_data_size(self.pathTest)
        self.data_loader = run_loader('test', self.pathTest, self.batchSize, self.imageSize, shuffle=False)
        self.test()

    def test(self):
        self.model.eval()
        acc = 0
        y_true = []
        y_hat = []
        for X, y in tqdm(self.data_loader):
            X = X.view(-1, self.imageSize * self.imageSize * 3)
            out = self.model(X)
            predictions = torch.argmax(out, 1)
            y_hat.append(predictions)
            y_true.append(y)
            acc += torch.sum(predictions == y).item()
        acc = acc / self.data_size
        y_hat = torch.cat(y_hat)
        y_true = torch.cat(y_true)
        print('Accuracy:', acc, '\n', 'Classification Report:', '\n', classification_report(y_hat, y_true))


if __name__ == "__main__":
    Testing("trained_f4/Model_195.model", "Data/monkey_species_test", 32, 32)
