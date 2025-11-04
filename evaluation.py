# evaluation.py

import torch
import torch.nn.functional as F
import PIL

def generate_caption(model, image_path, transform, tokenizer, max_seq_len=256, beam_size=3, device=torch.device("cpu"), print_process=False):
    """
    Generates a caption for a single image using beam search.
    """
    image = PIL.Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_output = model.encoder(image)
        beams = [([tokenizer.cls_token_id], 0)]
        completed = []

        for step in range(max_seq_len):
            new_beams = []
            for seq, score in beams:
                input_token = torch.tensor([seq]).to(device)
                target_mask = model.make_mask(input_token).to(device)
                pred = model.decoder(input_token, encoder_output, target_mask)
                pred = F.softmax(model.fc(pred), dim=-1)
                pred = pred[:, -1, :].view(-1)

                top_k_scores, top_k_tokens = pred.topk(beam_size)
                for i in range(beam_size):
                    new_seq = seq + [top_k_tokens[i].item()]
                    new_score = score + top_k_scores[i].item()
                    new_beams.append((new_seq, new_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            # Completed sequences
            for beam in beams[:]:
                if beam[0][-1] == tokenizer.sep_token_id:
                    completed.append(beam)
                    beams.remove(beam)
                    beam_size -= 1

            if beam_size == 0:
                break

        completed = completed or beams
        completed.sort(key=lambda x: x[1], reverse=True)
        best_seq = completed[0][0]
        return tokenizer.decode(best_seq, skip_special_tokens=True)
