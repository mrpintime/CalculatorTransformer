import torch
import torch.nn as nn
from tqdm import tqdm
from Data_Loader import causal_mask, get_ds 
import os
from pathlib import Path
from model import get_model 

import Configs


def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    sos_idx = tokenizer['[SOS]']
    eos_idx = tokenizer['[EOS]']

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encoder_p(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        # print("dec:===", decoder_input.shape)
        out = model.decoder_p(encoder_output=encoder_output, src_mask=source_mask, target_data=decoder_input, target_mask=decoder_mask)

        # get next token
        prob = model.proj(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)



def run_validation(model, validation_ds, tokenizer, max_len, device, print_msg, num_examples=2):
    tokenizer_id2text = {k:v for v,k in tokenizer.items()}
    model.eval()
    count = 0
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80
    device = 'cpu'
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = [tokenizer_id2text[t] for t in model_out.detach().cpu().numpy()]

            # Print the source, target and model output
            # print()
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break


def train_model(config, val=False, num_example=2):
    # Define the device
    # device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    print("Using device:", device)
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config=config, vocab_src_len=len(tokenizer.keys()), vocab_tgt_len=len(tokenizer.keys())).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = Configs.latest_weights_file_path(config) if preload == 'latest' else Configs.get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer['[PAD]'], label_smoothing=0.1).to(device)
    if val:
        batch_iterator = tqdm(val_dataloader, desc=f"Processing validation")
        run_validation(model, val_dataloader, tokenizer, config['seq_len_tgt'], device='cpu',
                       print_msg=lambda msg: batch_iterator.write(msg), num_examples=num_example)
    else:
        for epoch in range(initial_epoch, config['num_epochs']):
            torch.cuda.empty_cache()
            model.train()
            batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
            for batch in batch_iterator:

                encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
                decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

                # Run the tensors through the encoder, decoder and the projection layer
                encoder_output = model.encoder_p(encoder_input, encoder_mask) # (B, seq_len, d_model)
                decoder_output = model.decoder_p(encoder_output=encoder_output, src_mask=encoder_mask, target_data=decoder_input, target_mask=decoder_mask) # (B, seq_len, d_model)
                proj_output = model.proj(decoder_output) # (B, seq_len, vocab_size)
                

                # Compare the output with the label
                label = batch['label'].to(device) # (B, seq_len)

                # Compute the loss using a simple cross entropy
                loss = loss_fn(proj_output.view(-1, len(tokenizer.keys())), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
                
                # set gradient to zero
                optimizer.zero_grad(set_to_none=True)
                
                # Backpropagate the loss
                loss.backward()

                # Update the weights
                optimizer.step()

                global_step += 1

            # Run validation at the end of every 200 epoch
            if epoch % 200 == 0:
                run_validation(model, val_dataloader, tokenizer, config['seq_len_tgt'], device='cpu',
                       print_msg=lambda msg: batch_iterator.write(msg), num_examples=num_example)

        # Save the model at the end of every epoch
        model_filename = Configs.get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    config = Configs.get_config()
    train_model(config, val=True, num_example=10)
    