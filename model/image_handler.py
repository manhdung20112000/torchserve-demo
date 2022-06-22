from torchvision import transforms
from torch.profiler import ProfilerActivity
from ts.torch_handler.image_classifier import ImageClassifier

# ts (torchserve)


class MNISTClassifier(ImageClassifier):
    image_processing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    topk = 10

    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.profiler_args = {
            "activities": [ProfilerActivity.CPU],
            "record_shapes": True,
        }

    def postprocess(self, data):
        return data.argmax(1).tolist()
