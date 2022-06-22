# Serving pytorch model

# [Requirement](#requirement)
Torchserve requires Java JDK 11 while deploying
```bash
sudo apt install --no-install-recommends -y openjdk-11-jre-headless
```

Install python dependecies
```bash
pip install -r requirements.txt
```

# [Train and prepare model](#train-and-prepare-model)
Training model and save weight to '.pt' (or '.pth') file:
```python
# training
. . .
torch.save(model.state_dict(), 'path/to/file.pth')
```

For start training:
```bash
python train.py --lr 1e-4
```

# [Deployment](#deployment)
Firstly, we generate MAR file with **torch-model-archiver**:

```bash
torch-model-archiver --model-name mnistnet \
                     --version 1.0 \
                     --model-file model/model.py \
                     --serialized-file model/weights/net.pt \
                     --handler model/image_handler.py \
                     --extra-files model/index_to_labels.json
```

Note: 
- Torchserve provide have some default `handler`: *“image_classifier”, “image_segmenter”,  “object_detector”* and *“text_classifier”*.
- `--extra-file` extra dependency, this classification task is mapping dictionary from index to labels
- Model should define as a class store in .py file

**Deploy Torchserve**
Deploy the TorchServe REST APIs inclue Inference API, Management API and Metrics API. (They deployed on localhost in the port 8080, 8081, 8082 respectively and can change by manual through `config.propertise`).

```bash
torchserve --start \
           --ncs \
           --ts-config deployment/config.properties \
           --model-store deployment/model-store \
           --models mnistnet=mnistnet.mar
```
Check status (if server is 'Healthy')

```bash
curl http://localhost:8080/ping
```

# [Usage](#usage)
Request prediction through HTTP request's:

```bash
curl -X POST http://localhost:8080/predictions/mnistnet -T demo.png

# or

curl -X POST http://localhost:8080/predictions/mnistnet -T "data=@demo.png"
```