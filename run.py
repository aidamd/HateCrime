import json
import os
from MICNN import *
from Entity import Entity
from Hi_Attn import *

params = json.load(open("params.json", "r"))

if os.path.isfile("Data/" + params["dataset"] + "/data.pkl"):
    print("Loading data.pkl")
    data = pickle.load(open("Data/" + params["dataset"] + "/data.pkl", "rb"))
else:
    print("data.pkl not found. Generating bags ..")
    data = GenerateTrain(params["batch_size"], params["dataset"])

print("Loading train and test sets")
train_batches, dev_batches, test_batches, vocabs, embedding = data

if params["sub_task"] != "train":
    if os.path.isfile("Data/" + params["dataset"] + "/patch.pkl"):
        print("Loading unlabeled patch batches")
        unlabeled_batches = pickle.load(open("Data/" + params["dataset"] + "/patch.pkl", "rb"))
    else:
        print("Batching unlabeled patch data")
        unlabeled_batches = GenerateUnlabeled(vocabs, params["batch_size"], params["dataset"])

if params["task"] == "detect":
    if params["model"] == "MICNN":
        model = MICNN(params, vocabs, embedding)
    elif params["model"] == "Attn":
        model = Hi_Attn(params, vocabs, embedding)
    model.build()
    if params["sub_task"] == "train":
        train_sent, dev_sent, test_sent = model.run_model(
            BatchIt(train_batches, params["batch_size"], vocabs),
            BatchIt(dev_batches, params["batch_size"], vocabs),
            BatchIt(test_batches, params["batch_size"], vocabs))
        if train_sent:
            for i in range(len(train_batches)):
                tmp = list(train_batches[i])[:6]
                tmp.append(list(train_sent[i]))
                train_batches[i] = tuple(tmp)
            for i in range(len(dev_batches)):
                tmp = list(dev_batches[i])[:6]
                tmp.append(list(dev_sent[i]))
                dev_batches[i] = tuple(tmp)
            for i in range(len(test_batches)):
                tmp = list(test_batches[i])[:6]
                tmp.append(list(test_sent[i]))
                test_batches[i] = tuple(tmp)
            pickle.dump((train_batches, dev_batches, test_batches,
                         vocabs, embedding), open("Data/" + params["dataset"] + "/data.pkl", "wb"))
    elif params["sub_task"] == "predict":
        hate_pred, indice_pred = model.predict(unlabeled_batches)
        pickle.dump(hate_pred, open("Data/" + params["dataset"] + "/labels.pkl", "wb"))
        hate_batches = list()
        for i in range(len(unlabeled_batches)):
            if hate_pred[i] == 1:
                if params["model"] == "MICNN":
                    hate_batches.append(unlabeled_batches[i] + (hate_pred[i], indice_pred[i], i))
                else:
                    hate_batches.append(unlabeled_batches[i] + (hate_pred[i], i))
        pickle.dump(hate_batches, open("Data/" + params["dataset"] + "/predict.pkl", "wb"))
    elif params["sub_task"] == "active":
        probabilities = model.active_learn(unlabeled_batches)
        # pickle.dump(hate_pred, open("Data/patch_hate.pkl", "wb"))
        pickle.dump(probabilities, open("probability.pkl", "wb"))


if params["task"] == "extract":
    hate_train_batches = [train for train in train_batches if train[2] == 1]
    hate_dev_batches = [dev for dev in dev_batches if dev[2] == 1]
    hate_test_batches = [test for test in test_batches if test[2] == 1]

    t_weights = np.array([1 - (Counter([train[3] for train in hate_train_batches])[i] /
                               len(hate_train_batches)) for i in range(8)])
    a_weights = np.array([1 - (Counter([train[4] for train in hate_train_batches])[i] /
                               len(hate_train_batches)) for i in range(4)])
    entity = Entity(params, vocabs, embedding)
    entity.build()
    if params["sub_task"] == "train":
        entity.run_model(BatchIt(hate_train_batches, params["batch_size"], vocabs),
                     BatchIt(hate_dev_batches, params["batch_size"], vocabs),
                     BatchIt(hate_test_batches, params["batch_size"], vocabs),
                     (t_weights, a_weights))
    elif params["sub_task"] == "predict":
        unlabeled_batches = pickle.load(open("Data/" + params["dataset"] + "/predict.pkl", "rb"))
        target, action = entity.predict(unlabeled_batches, (t_weights, a_weights))
        pickle.dump(target, open("Data/" + params["dataset"] + "/targets.pkl", "wb"))
        pickle.dump(action, open("Data/" + params["dataset"] + "/actions.pkl", "wb"))
