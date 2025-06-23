import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Wav2Vec2ConformerForCTC
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig

# Check if CUDA is available
device_audio = torch.device("cuda:0")  # Device for the audio encoder
device_llm = torch.device("cuda:1")    # Device for the LLM

# 1. Load the LLaMA model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Configure the model for 4-bit quantization
bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16
}

# Load the LLM explicitly on device 1
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    use_cache=False
).to(device_llm)

# 2. Configure QLoRA with PEFT (Parameter-Efficient Fine-Tuning)
lora_config = LoraConfig(
    r=16,  # Low-rank dimension
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.01,  # Dropout for regularization
    task_type="CAUSAL_LM"  # Task type for causal language modeling
)
model = prepare_model_for_kbit_training(model)
llm_model = get_peft_model(model, lora_config).to(device_llm)  # Explicitly place on device 1


# 3. Define the Audio Encoder with Downsampling and Conformer
class AudioEncoder(nn.Module):
    def __init__(self, input_dim, conformer_hidden_dim, pretrained_model_name="facebook/wav2vec2-conformer-rope-large-960h-ft"):
        """
        Audio encoder with downsampling layers and Conformer in between.
        :param input_dim: Dimension of the input audio features.
        :param conformer_hidden_dim: Hidden dimension of the Conformer model.
        :param pretrained_model_name: Pretrained Wav2Vec2Conformer model name.
        """
        super(AudioEncoder, self).__init__()

        # First downsampling block
        self.downsample1 = nn.Sequential(
            nn.Linear(input_dim, conformer_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Conformer block
        self.conformer = Wav2Vec2ConformerForCTC.from_pretrained(pretrained_model_name)

        # Second downsampling block
        self.downsample2 = nn.Sequential(
            nn.Linear(conformer_hidden_dim, conformer_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, audio_features, attention_mask=None):
        """
        Process raw audio features into embeddings.
        :param audio_features: [batch_size, seq_len, input_dim]
        :param attention_mask: Optional attention mask for padded audio.
        :return: [batch_size, seq_len, final_dim] - Audio embeddings.
        """
        # Move inputs to the correct device
        audio_features = audio_features.to(device_audio)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device_audio)

        # First downsampling block
        x = self.downsample1(audio_features)

        # Conformer block
        outputs = self.conformer(x, attention_mask=attention_mask, output_hidden_states=True)
        x = outputs.hidden_states[-1]  # Extract the last hidden state

        # Second downsampling block
        x = self.downsample2(x)

        return x


# 4. Define the Disfluency Model
class DisfluencyModel(nn.Module):
    def __init__(self, audio_encoder, llama_model):
        """
        Combined model for disfluency detection.
        :param audio_encoder: Audio encoder with downsampling and Conformer.
        :param llama_model: Pre-trained LLM (e.g., LLaMA).
        """
        super(DisfluencyModel, self).__init__()
        self.audio_encoder = audio_encoder  # Processes raw audio into embeddings
        self.llm = llama_model  # Pre-trained LLM

    def forward(self, audio_features, input_ids, attention_mask):
        """
        Forward pass for the model.
        :param audio_features: Raw audio features [batch_size, seq_len].
        :param input_ids: Tokenized textual input for LLM [batch_size, text_seq_len].
        :param attention_mask: Attention mask for text tokens [batch_size, text_seq_len].
        :param loss_mask: Mask for excluding prompt tokens from loss calculation.
        :return: Logits from the LLM.
        """
        # Step 1: Extract audio embeddings from the audio encoder
        audio_embeddings = self.audio_encoder(audio_features.to(device_audio))  # [batch_size, audio_seq_len, hidden_dim]

        # Step 2: Extract token embeddings from the LLM's embedding layer
        token_embeddings = self.llm.get_input_embeddings()(input_ids.to(device_llm))  # [batch_size, text_seq_len, hidden_dim]

        # Step 3: Concatenate audio and text embeddings along the sequence dimension
        combined_embeddings = torch.cat((audio_embeddings.to(device_llm), token_embeddings), dim=1)  # [batch_size, combined_seq_len, hidden_dim]

        # Step 4: Create a unified attention mask
        audio_seq_len = audio_embeddings.size(1)
        text_seq_len = token_embeddings.size(1)
        audio_attention_mask = torch.ones((audio_embeddings.size(0), audio_seq_len), device=device_llm)  # All ones for audio
        combined_attention_mask = torch.cat((audio_attention_mask, attention_mask.to(device_llm)), dim=1)  # [batch_size, combined_seq_len]

        # Step 5: Forward pass through the LLM with concatenated embeddings
        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask
        )

        return logits


def create_disfluency_model(input_dim, conformer_hidden_dim, final_dim):
    """
    Factory function to initialize the DisfluencyModel.
    :param input_dim: Dimension of the raw audio input features.
    :param conformer_hidden_dim: Hidden dimension of the Conformer.
    :param final_dim: Final embedding dimension for the LLM.
    :return: DisfluencyModel instance.
    """
    # Initialize the audio encoder
    audio_encoder = AudioEncoder(
        input_dim=input_dim,
        conformer_hidden_dim=conformer_hidden_dim,
        pretrained_model_name="facebook/wav2vec2-conformer-rope-large-960h-ft"
    ).to(device_audio)

    # Initialize the combined model
    model = DisfluencyModel(audio_encoder, llm_model)
    return model
