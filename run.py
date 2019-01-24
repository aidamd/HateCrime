import json
from MICNN import *
from Entity import Entity
from Hi_Attn import *

params = json.load(open("params.json", "r"))

if os.path.isfile("Data/data.pkl"):
    print("Loading data.pkl")
    data = pickle.load(open("Data/data.pkl", "rb"))
else:
    print("data.pkl not found. Generating bags ..")
    data = GenerateTrain(params["batch_size"])

print("Loading train and test sets")
train_batches, dev_batches, test_batches, vocabs, embedding = data

if not params["train"]:
    if os.path.isfile("Data/patch.pkl"):
        print("Loading unlabeled patch batches")
        unlabeled_batches = pickle.load(open("Data/patch.pkl", "rb"))
    else:
        print("Batching unlabeled patch data")
        unlabeled_batches = GenerateUnlabeled(vocabs, params["batch_size"])

else:
    unlabeled_batches = []

if "detect" in params["task"]:
    if params["model"] == "MICNN":
        model = MICNN(params, vocabs, embedding)
    elif params["model"] == "Attn":
        model = Hi_Attn(params, vocabs, embedding)
    model.build()
    predictions = model.run_model(BatchIt(train_batches, params["batch_size"], vocabs),
                                  BatchIt(dev_batches, params["batch_size"], vocabs),
                                  BatchIt(test_batches, params["batch_size"], vocabs),
                                  unlabeled_batches)
    pickle.dump(predictions, open("Data/patch_hate.pkl", "wb"))

if "extract" in params["task"]:
    train_batches = [train for train in train_batches if train[2] == 1]
    dev_batches = [dev for dev in dev_batches if dev[2] == 1]
    t_weights = np.array([1 - (Counter([train[3] for train in train_batches])[i] /
                               len(train_batches)) for i in range(8)])
    a_weights = np.array([1 - (Counter([train[4] for train in train_batches])[i] /
                               len(train_batches)) for i in range(4)])
    entity = Entity(params, vocabs, embedding)
    entity.build()
    entity.run_model(BatchIt(train_batches, params["batch_size"], vocabs),
                     BatchIt(dev_batches, params["batch_size"], vocabs),
                     unlabeled_batches, (t_weights, a_weights))
