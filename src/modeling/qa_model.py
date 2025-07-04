import torch
import torch.nn as nn
from transformers import XLMRobertaForQuestionAnswering

########################################
# Model Definition
########################################

class QAWithYesNoModel(XLMRobertaForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)
        # One classification head with 2 outputs:
        # output[0]: is_bool logit (0: extractive, 1: boolean)
        # output[1]: bool_val logit (0 for False, 1 for True)
        self.boolean_classifier = nn.Linear(config.hidden_size, 2)  # is_bool, bool_value

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        start_positions=None, 
        end_positions=None, 
        is_bool_labels=None,   # Tensor with 0 (extractive) or 1 (boolean)
        bool_val_labels=None,  # Tensor with 0 (False) or 1 (True) for boolean examples; dummy for extractive examples.
#         **kwargs  # Accept any extra keyword arguments without error.
    ):
        
#         # Pop our extra keys so they don't get passed to the parent's forward.
#         is_bool_labels = kwargs.pop("is_bool_labels", None)
#         bool_val_labels = kwargs.pop("bool_val_labels", None)
        
        # Now call the parent's forward method with only the remaining keys.
        outputs = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            start_positions=start_positions,
            end_positions=end_positions,
            output_hidden_states=True,
#              **kwargs
        )
        
        # Extract the [CLS] token representation from the last hidden layer.
        sequence_output = outputs.hidden_states[-1][:, 0, :] # shape: (batch_size, hidden_size)
        
         # One linear layer produces 2 logits per example.
        classification_logits = self.boolean_classifier(sequence_output)  # shape: (batch_size, 2)
        
        # Separate the outputs for clarity.
        is_bool_logits = classification_logits[:, 0]    # Logit for is_bool
        bool_val_logits = classification_logits[:, 1]    # Logit for bool value
        
        total_loss = outputs.loss  # start with the QA loss
        
        if (is_bool_labels is not None) and (bool_val_labels is not None):
            loss_fct = nn.BCEWithLogitsLoss()
            # Loss for is_bool prediction across the full batch.
            is_bool_loss = loss_fct(is_bool_logits, is_bool_labels.float())
            
            # For the boolean value loss, compute only on the boolean examples
            mask = (is_bool_labels == 1)
            if mask.sum() > 0:
                bool_val_loss = loss_fct(bool_val_logits[mask], bool_val_labels.float()[mask])
            else:
                bool_val_loss = 0.0
            
            total_loss = total_loss + is_bool_loss + bool_val_loss

        return {
            "loss": total_loss, 
            "start_logits": outputs.start_logits, 
            "end_logits": outputs.end_logits, 
            "is_bool_logits": is_bool_logits.unsqueeze(-1),
            "bool_val_logits": bool_val_logits.unsqueeze(-1),
        }