"""
llm_utils.py
Utility functions for querying LLMs and tokenizing queries.
"""
from io import BytesIO
import torch
import os
import base64
from openai import OpenAI
from qwen_vl_utils import process_vision_info

def query_to_tokenized_query(input_text, t5_model, t5_tokenizer, max_length=50):
    """Tokenize a query using a T5 model and tokenizer."""
    inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=64, padding="max_length", truncation=True)
    with torch.no_grad():
        output_ids = t5_model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
    output_text = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def ask_llm(prompt, image, config, detection_2d=None, vlm_model=None, vlm_processor=None):
    """Query an LLM (Qwen or GPT-4) with a prompt and image."""
    if not config.query_tokenization:
        if config.tracking_type == "Car":
            with open("dataset/prompts/prompt.txt", "r") as f:
                sysprompt = f.read()
        else:
            with open("dataset/prompts/pedestrian_prompt.txt", "r") as f:
                sysprompt = f.read()
    else:
        with open("dataset/prompts/prompt_tokenization.txt", "r") as f:
            sysprompt = f.read()

    if detection_2d is not None:
        # Ensure detection_2d is a 1D array/list of 4 elements
        if hasattr(detection_2d, 'shape') and detection_2d.shape == (4,):
            x1, y1, x2, y2 = detection_2d.tolist()
        elif isinstance(detection_2d, (list, tuple)) and len(detection_2d) == 4:
            x1, y1, x2, y2 = detection_2d
        elif hasattr(detection_2d, 'shape') and len(detection_2d.shape) == 2 and detection_2d.shape[0] == 1 and detection_2d.shape[1] == 4:
            x1, y1, x2, y2 = detection_2d[0].tolist()
        else:
            raise ValueError(f"Unexpected detection_2d shape: {getattr(detection_2d, 'shape', type(detection_2d))}")
        cropped_image = image.crop((x1, y1, x2, y2))
    else:
        cropped_image = image
    ## LAVA 
    if config.vlm_model == "llava":
        if vlm_model is None or vlm_processor is None:
            raise ValueError("qwen_model and qwen_processor must not be None when using 'llava' as vlm_model. Please ensure they are loaded and passed correctly.")
        sysprompt += f'\n Here below is the input you should process. Replace the TOKEN> tokens based on the coordinates instruction, filter out the rest. If you need to specify a COLOR>, specify if the COLOR is light or dark from the picture. Dont mix POSITION and DIRECTION informations.'
        sysprompt += f'\n INPUT: {prompt}'
        prompt = f"USER: <image>\n {sysprompt} ASSISTANT:"
        inputs = vlm_processor(images=cropped_image, text=prompt, return_tensors="pt")
        device = next(vlm_model.parameters()).device
        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(device)
        generate_ids = vlm_model.generate(**inputs, max_new_tokens=25)
        full_answer = vlm_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        answer = full_answer.split("ASSISTANT:")[1].strip()
        answer = answer.replace(">", "")
    ## QWEN
    elif config.vlm_model == "qwen":
        if vlm_model is None or vlm_processor is None:
            raise ValueError("qwen_model and qwen_processor must not be None when using the vlm_model. Please ensure they are loaded and passed correctly.")
        sysprompt += f'\n Here below is the input you should process. Replace the TOKEN> tokens based on the coordinates instruction, filter out the rest. If you need to specify a COLOR>, specify if the COLOR is light or dark from the picture. Dont mix POSITION and DIRECTION informations.'
        sysprompt += f'\n INPUT: {prompt} \n Assistant:'
        # Qwen2-VL uses specific image token format with process_vision_info
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": cropped_image},
                    {"type": "text", "text": sysprompt}
                ]
            }
        ]
        
        # Use process_vision_info to properly format the image
        image_inputs, _ = process_vision_info(messages)
        prompt_text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = vlm_processor(text=[prompt_text], images=image_inputs, return_tensors="pt", padding=True)
        device = next(vlm_model.parameters()).device
        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(device)

        prompt_length = inputs['input_ids'].shape[1]
        generate_ids = vlm_model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        # Debug: Check the full generated output
        full_answer = vlm_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # Decode only the generated part (excluding the prompt) 
        generated_tokens = generate_ids[:, prompt_length:]
        answer = vlm_processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        answer = answer.strip()
        answer = answer.replace(">", "")
    ## PHI
    elif config.vlm_model == "phi":
        if vlm_model is None or vlm_processor is None:
            raise ValueError("phi_model and phi_processor must not be None when using the vlm_model. Please ensure they are loaded and passed correctly.")
        sysprompt += f'\n Here below is the input you should process. Replace the TOKEN> tokens based on the coordinates instruction, filter out the rest. If you need to specify a COLOR>, specify if the COLOR is light or dark from the picture. Dont mix POSITION and DIRECTION informations.'
        sysprompt += f'\n INPUT: {prompt}'
        prompt = f"<|user|>\n<|image_1|>\n{sysprompt}<|end|>\n<|assistant|>\n"
        inputs = vlm_processor(text=prompt, images=[cropped_image], return_tensors="pt")
        device = next(vlm_model.parameters()).device
        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(device)

        prompt_length = inputs['input_ids'].shape[1]
        generate_ids = vlm_model.generate(**inputs, max_new_tokens=25, do_sample=False)
        answer = vlm_processor.decode(generate_ids[0][prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        answer = answer.strip()
        answer = answer.replace(">", "")
    ## GPT
    elif config.vlm_model == "gpt4":
        buffer = BytesIO()
        cropped_image.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode("utf-8")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key == "":
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set your OpenAI API key.")
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sysprompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]}
                ],
            )
            answer = response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error communicating with OpenAI API: {e}\nCheck your API key and network connection.")
    else:
        answer = None
    return answer
