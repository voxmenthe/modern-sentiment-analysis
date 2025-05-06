
# Example using ModernBERT for multi-label sequence classification
import torch
from transformers import AutoTokenizer, ModernBertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", problem_type="multi_label_classification")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = ModernBertForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base", num_labels=num_labels, problem_type="multi_label_classification"
)

labels = torch.sum(
    torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
).to(torch.float)
loss = model(**inputs, labels=labels).loss


# Subclassing a pretrained model for a new objective
# 🤗Transformers
# 3.5k views


erickrf
Oct 2021
I would like to use a pretrained model as an encoder for a new task. It is essentially multiple sequence classification objectives, like in the ...ForSequenceClassification models, but with an output layer for each subtask.

I could just create wrappers around the encoder, but I’d like to subclass from PreTrainedModel to better integrate with the Trainer class. How exactly should I do? Do I need to create a config class as well? I will at least need to supply an extra list or dict to the config telling how many classes each subtask has.

Thanks!


sgugger
Oct 2021
You can definitely subclass PretrainedConfig for your custom config and PreTrainedModel for your custom model, then access all the methods of the library.


erickrf
Oct 2021
@sgugger thanks! But in that case what is needed to make methods like from_pretrained work out of the box? I saw that the pretrained model classes have a class attribute called config_class, is setting that enough?



sgugger
Oct 2021
It’s to find the right config in the Transformers library. In your case, you might have to use two steps:

config = CustomConfig.from_pretrained(path_to_folder_with_config_and_weights)
model = CustomModel.from_pretrained(path_to_folder_with_config_and_weights, config)


erickrf
Oct 2021
Ok. But how can I load the pretrained model (i.e., the encoder inside my class)?
I tried doing CustomModel.from_pretrained(path_to_pretrained, additional_config_data), but that ignored all the weights in the checkpoint (name mismatches, I suppose?).


sgugger
Oct 2021
Did you save the corresponding model with save_pretrained?


erickrf
Oct 2021
Nope, I haven’t even fine tuned the model yet :slight_smile:
I’m calling from_pretrained in the encoder directly, after creating the classifier object and before training, but that looks hacky.


sgugger
Oct 2021
I’m not sure what you want to do, but calling from_pretrained on your class with weights saved from another class will not work. If you want to use a pretrained model for part of the your custom model, you should use the from_pretrained method when defining that part of your custom model.