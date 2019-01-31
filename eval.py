import torch
from model import FC4
from PIL import Image
from load import load_data_test


class Classifier:
    def __init__(self, model_path, i2class, img_size):
        self.img_size = img_size
        self._load_model(model_path)
        self.transformer = load_data_test(32)
        self.i2class = i2class

    def _load_model(self, model_path):
        model = FC4()
        state_dict = torch.load(model_path) 
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

    def predict(self, path_img):
        out = self.load_image(path_img)
        prediction = torch.argmax(out, 1).item()
        return self.i2class[prediction]

    def predict_proba(self, path_img):
        dictProba = {}
        out = self.load_image(path_img)
        probabilities = torch.nn.functional.softmax(out, dim=1)
        for i, item in enumerate(self.i2class):
            dictProba[item] = torch.tensor(probabilities)[0][i]
        return dictProba

    def load_image(self, path_img):
        img = Image.open(path_img)
        img = self.transformer(img)
        img = img.view(1, self.img_size * self.img_size * 3)
        return self.model(img)


if __name__ == "__main__":
    i = Classifier('monkey_species/trained_f4/Model_195.model', ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9'], 32)
    print(i.predict_proba('eval_data/n8021.jpg'))
    print(i.predict('eval_data/n8021.jpg'))
