from flask import Flask, request, jsonify
from typing import Any, List, cast
import h5py
import structlog
from colpali_engine.utils.torch_utils import get_torch_device
from peft import LoraConfig
from PIL import Image
from colqwen2 import ColQwen2ForRAG
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import torch
import os
from pdf2image import convert_from_path
import traceback
################### Setup logger ############################
structlog.configure(
    processors=[
        structlog.processors.KeyValueRenderer(key_order=["event", "user", "status"])
    ]
)

logger = structlog.get_logger()


################### Setup Model and Processor ############################

model_name = "vidore/colqwen2-v1.0"
device = get_torch_device("auto")

logger.warn(f"Using device: {device}")
# Get the LoRA config from the pretrained retrieval model
lora_config = LoraConfig.from_pretrained(model_name)

# Load the processors
processor_retrieval = cast(ColQwen2Processor, ColQwen2Processor.from_pretrained(model_name))
processor_generation = cast(Qwen2VLProcessor, Qwen2VLProcessor.from_pretrained(lora_config.base_model_name_or_path))

# Load the model with the loaded pre-trained adapter for retrieval
model = cast(
    ColQwen2ForRAG,
    ColQwen2ForRAG.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ),
)

################### Setup Flask app ############################
app = Flask(__name__)
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT", 125))
RETRIEVE_K = int(os.environ.get("RETRIEVE_K", 1))

def process_pdf(file_path):
    """
    Process a PDF file by extracting its pages and running them through ColQwen2 for retrieval.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        torch.Tensor: The embeddings of the pages in the PDF as a tensor with shape (num_pages, embedding_dim).
    """
    images = convert_from_path(file_path)
    images = [scale_image(image, new_height=IMAGE_HEIGHT) for image in images]
    
    model.enable_retrieval()
    data = processor_retrieval.process_images(images)
    with torch.no_grad():
        batch_doc = {k: v.to(model.device) for k, v in data.items()}
        embeddings_doc = model(**batch_doc)
    return images, embeddings_doc.cpu().float().numpy()

def process_query(query):
    """
    Process a query by running it through ColQwen2 for retrieval.

    Args:
        query (str): The query string.

    Returns:
        torch.Tensor: The embedding of the query as a tensor with shape (1, embedding_dim).
    """
    data = processor_retrieval.process_queries([query])    
    model.enable_retrieval()
    with torch.no_grad():
        batch_doc = {k: v.to(model.device) for k, v in data.items()}
        embeddings_doc = model(**batch_doc)
    return embeddings_doc.cpu().float().numpy()

def get_data_by_filename(filename):
    """
    Retrieve vector and image data from the knowledge base HDF5 file for a given filename.

    Args:
        filename (str): The name of the file to look up in the HDF5 knowledge base.

    Returns:
        tuple: A tuple containing the vector data and image data if the filename is found,
               or None if the filename is not present in the knowledge base.
    """

    with h5py.File(os.path.join('data','knowledge_base.h5'), 'r') as f:
        if filename in f:
            return f[filename]['vector'][()], f[filename]['image'][()]
        else:
            raise ValueError(f"Filename {filename} not found in knowledge base.")

def scale_image(image: Image.Image, new_height: int = 1024) -> Image.Image:
    """
    Scale an image to a new height while maintaining the aspect ratio.
    """
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    scaled_image = image.resize((new_width, new_height))

    return scaled_image

def store_vector(filename, vector_data, image_data):
    """
    Stores the vector data for a given filename in an HDF5 file called 'vectors.h5'.

    Args:
        filename (str): The filename of the PDF file.
        vector_data (list): The vector data to store.

    Returns:
        None
    """
    os.makedirs('data', exist_ok=True)
    with h5py.File(os.path.join('data','knowledge_base.h5'), 'a') as f:
        # Create a group for each filename if it doesn't exist
        if filename not in f:
            group = f.create_group(filename)
        else:
            group = f[filename]
        
        # Store the vector data
        if 'vector' in group:
            del group['vector']  # Delete existing dataset if it exists
        
        # Store the vector data
        if 'image' in group:
            del group['image']  # Delete existing dataset if it exists
        group.create_dataset('vector', data=vector_data)
        group.create_dataset('image', data=image_data)
    logger.info(f"Stored vector data for {filename} in knowledge base.")
    
@app.route('/upload', methods=['POST'])
def upload_pdf():
    """
    Handles POST requests to upload a PDF file. The file is saved to the uploads folder, processed with ColQwen2 for retrieval, and the resulting embeddings are stored in an HDF5 file called knowledge_base.h5.

    Args:
        None

    Returns:
        A JSON response with a status message of "Document processed".
    """
    try:
        os.makedirs('uploads', exist_ok=True)
        file = request.files['pdf']
        filename = file.filename
        file.save(os.path.join('uploads', filename))
        
        # Process with ColQwen2 retrieval model
        images, page_embeddings = process_pdf(os.path.join('uploads', filename))
        
        # Store in HDF5
        store_vector(filename, page_embeddings, images)
        return jsonify({"status": "Document processed", "code": 200})
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return jsonify({"status": "Error processing document", "error": str(e), "code": 500, "trace": traceback.format_exc()})
    
def generate_answer(query, images):
    """
    Generates an answer to a given query using input images and a Retrieval-Augmented Generation (RAG) model.

    Args:
        query (str): The question or query that needs to be answered.
        images (torch.Tensor): The tensor containing image data to be used for answering the query.

    Returns:
        List[str]: A list of generated text responses based on the input query and images.
    """

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": f"Answer the following question using the input image: {query}",
                },
            ],
        }
    ]
    text_prompt = processor_generation.apply_chat_template(conversation, add_generation_prompt=True)
    inputs_generation = processor_generation(
        text=[text_prompt],
        images=[images],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generate the RAG response
    model.enable_generation()
    output_ids = model.generate(**inputs_generation, max_new_tokens=100)

    # Ensure that only the newly generated token IDs are retained from output_ids
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs_generation.input_ids, output_ids)]

    # Decode the RAG response
    output_text = processor_generation.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return output_text
    
    
@app.route('/query', methods=['POST'])
def handle_query():
    """
    Handles a POST request to the '/query' endpoint that contains a filename and a query.
    Retrieves the corresponding image data from the HDF5 file and passes it to the
    generate_answer function to generate a text response based on the query and the
    retrieved image.
    """
    data = request.json
    filename = data['filename']
    query = data['query']
    # Retrieve with adapter switching
    query_embedding = process_query(query)
    
    image_embedding, images = get_data_by_filename(filename)
    
    # Compute scores between the query embedding and the image embeddings
    scores = processor_retrieval.score_multi_vector(
        torch.from_numpy(query_embedding), 
        torch.from_numpy(image_embedding)
    )
    
    # Get the index of the top-scoring page
    top_pages = scores.numpy()[0].argsort()[-RETRIEVE_K:][::-1]
    
    # Generate an answer based on the query and the top-scoring page
    answer = generate_answer(query, images[top_pages][0])
    
    return jsonify({"query": query, "answer": answer[0]})

@app.route('/filenames', methods=['GET'])
def get_filenames():
    """
    Handles a GET request to the '/filenames' endpoint that returns a list of filenames
    present in the knowledge base HDF5 file.
    """
    with h5py.File(os.path.join('data','knowledge_base.h5'), 'r') as f:
        filenames = list(f.keys())
    return jsonify({"filenames": filenames})

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)