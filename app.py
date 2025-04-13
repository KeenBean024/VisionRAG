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
import numpy as np
import cv2
import base64
import gc
from io import BytesIO
from tqdm import tqdm
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
torch.cuda.empty_cache()

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
model.eval()
################### Setup Flask app ############################
app = Flask(__name__)
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT", 125))
RETRIEVE_K = int(os.environ.get("RETRIEVE_K", 1))
SCALE_IMAGE = os.environ.get("SCALE_IMAGE", "true").lower() == "true"
KNOWLEDGE_BASE = os.path.join('/data', "knowledge_base.h5")
ROW_SIZE = 128  # The last dimension is fixed

# Define a fixed-size array dtype for a row of embedding (128 float32s)
row_dtype = np.dtype((np.float32, (ROW_SIZE,)))
# Create a variable-length dtype based on that fixed-size type:
vlen_float_dtype = h5py.special_dtype(vlen=np.dtype('float32'))


logger.warn(f"Using image scaling: {SCALE_IMAGE}")
logger.warn(f"Using retrieve k: {RETRIEVE_K}")
logger.warn(f"Using image height: {IMAGE_HEIGHT}")

def process_pdf(file_path):
    """
    Process a PDF file page by page, extract embeddings, and store embeddings and images in HDF5.
    Each page's embedding is expected to have shape (1, N, 128); after dropping the batch dimension,
    we get (N, 128). We then flatten it to a 1D array (of length N*128) and store the number N in a separate dataset.
    """
    print("Starting PDF processing...")
    images_gen = convert_from_path(file_path)

    with h5py.File(KNOWLEDGE_BASE, 'a') as f:
        filename = os.path.basename(file_path)
        if filename in f:
            del f[filename]
        group = f.create_group(filename)

        # Create extendable dataset for flattened embeddings.
        vector_dataset = group.create_dataset(
            "vectors",
            shape=(0,),
            maxshape=(None,),
            dtype=vlen_float_dtype,
            compression="gzip"
        )
        # Dataset to store the number of rows (N) for each page's embedding.
        shape_dataset = group.create_dataset(
            "vector_shapes",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            compression="gzip"
        )
        # Create extendable dataset for images as variable-length arrays of bytes.
        image_dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
        image_dataset = group.create_dataset(
            "images",
            shape=(0,),
            maxshape=(None,),
            dtype=image_dtype
        )
        model.enable_retrieval()

        for i, image in tqdm(enumerate(images_gen), desc="Processing pages"):
            if SCALE_IMAGE:
                image = scale_image(image, new_height=IMAGE_HEIGHT)

            data = processor_retrieval.process_images([image])
            with torch.no_grad():
                batch_doc = {k: v.to(model.device) for k, v in data.items()}
                embedding = model(**batch_doc).cpu().float().numpy()
            # Drop the redundant batch dimension: expected shape becomes (N, 128)
            embedding = embedding[0]
            # print(f"Embedding shape - {embedding.shape}")  # e.g. (779, 128)

            # Flatten the embedding: (N, 128) -> (N*128,)
            flat_embedding = embedding.flatten()

            # Convert image to compressed JPEG bytes
            img_buffer = BytesIO()
            image.save(img_buffer, format="JPEG", quality=75)
            img_bytes = np.frombuffer(img_buffer.getvalue(), dtype=np.uint8)

            # Extend datasets dynamically
            vector_dataset.resize((i + 1,))
            shape_dataset.resize((i + 1,))
            image_dataset.resize((i + 1,))

            vector_dataset[i] = flat_embedding.tolist()
            shape_dataset[i] = embedding.shape[0]  # number of rows N
            image_dataset[i] = img_bytes

            # Cleanup
            del data, batch_doc, embedding, flat_embedding, image, img_buffer, img_bytes
            torch.cuda.empty_cache()
            gc.collect()

    print(f"PDF processing complete. Data stored in group '{filename}'")

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

def process_image(image):
    """
    Process a image by running it through ColQwen2 for retrieval.

    Args:
        image : The image.

    Returns:
        torch.Tensor: The embedding of the image as a tensor with shape (1, embedding_dim).
    """
    data = processor_retrieval.process_images(image)
    model.enable_retrieval()
    with torch.no_grad():
        batch_doc = {k: v.to(model.device) for k, v in data.items()}
        embeddings_doc = model(**batch_doc)
    return embeddings_doc.cpu().float().numpy()

def get_embeddings_by_filename(filename):
    with h5py.File(KNOWLEDGE_BASE, 'r') as f:
        if filename not in f:
            raise ValueError(f"Filename {filename} not found in knowledge base.")
        group = f[filename]
        vectors = group['vectors']
        shapes = group['vector_shapes']
        embeddings = []
        # Reconstruct each page's embedding from the flattened data.
        for i in range(vectors.shape[0]):
            flat_embedding = np.array(vectors[i], dtype=np.float32)
            # Retrieve the number of rows (N) stored for this page.
            N = int(shapes[i])
            # Reshape flat_embedding (of length N*128) to (N, 128)
            embedding = flat_embedding.reshape(N, ROW_SIZE)
            embeddings.append(embedding)
        return np.array(embeddings)

def get_image_by_filename(filename, page_index):
    with h5py.File(KNOWLEDGE_BASE, 'r') as f:
        if filename not in f:
            raise ValueError(f"Filename {filename} not found in knowledge base.")
        # Retrieve image bytes for the specified page.
        img_bytes = bytes(f[filename]['images'][page_index])
        return Image.open(BytesIO(img_bytes))

def scale_image(image: Image.Image, new_height: int = 1024) -> Image.Image:
    """
    Scale an image to a new height while maintaining the aspect ratio.
    """
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    scaled_image = image.resize((new_width, new_height))

    return scaled_image


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
        process_pdf(os.path.join('uploads', filename))
        
        # Store in HDF5
        return jsonify({"status": "Document processed"}), 200
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return jsonify({"status": "Error processing document", "error": str(e), "trace": traceback.format_exc()}), 500
    finally:
        try:
            del images, page_embeddings
        except:
            pass
        torch.cuda.empty_cache()
        
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
                *[
                    {"type": "image"}
                    for _ in images
                ],
                {
                    "type": "text",
                    "text": f"You are a Question answering bot. Using the multiple image input as context, answer the following QUESTION: {query}",
                    # "text": f"You are an intelligent reasoning assistant. Analyze the provided images, extract relevant data, and apply logical step-by-step reasoning to answer the QUESTION accurately. For math problems, show clear intermediate steps before giving the final answer. Now, answer: {query}",
                },
            ],
        }
    ]
    text_prompt = processor_generation.apply_chat_template(conversation, add_generation_prompt=True)
    inputs_generation = processor_generation(
        text=[text_prompt],
        images=images,
        padding=True,
        max_length=4096,
        return_tensors="pt",
    ).to(device)

    # Generate the RAG response
    model.enable_generation()
    output_ids = model.generate(**inputs_generation, max_new_tokens=512, early_stopping=False,  do_sample=True)

    # Ensure that only the newly generated token IDs are retained from output_ids
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs_generation.input_ids, output_ids)]

    # Decode the RAG response
    output_text = processor_generation.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    del inputs_generation, output_ids, generated_ids
    torch.cuda.empty_cache()
    
    return output_text
    
    
@app.route('/query', methods=['POST'])
def handle_query():
    """
    Handles a POST request to the '/query' endpoint that contains a filename and a query.
    Retrieves the corresponding image data from the HDF5 file and passes it to the
    generate_answer function to generate a text response based on the query and the
    retrieved image.
    """
    try:
        data = request.json
        filename = data['filename']
        query = data['query']
        # Retrieve with adapter switching
        query_embedding = process_query(query)
        
        pdf_embedding = get_embeddings_by_filename(filename)
        
        # Compute scores between the query embedding and the image embeddings
        scores = processor_retrieval.score_multi_vector(
            torch.from_numpy(query_embedding), 
            torch.from_numpy(pdf_embedding)
        )
        
        # Get the index of the top-scoring page
        top_pages = scores.numpy()[0].argsort()[-RETRIEVE_K:][::-1]
        images = [get_image_by_filename(filename, page) for page in top_pages]
        # Generate an answer based on the query and the top-scoring page
        answer = generate_answer(query, images)
        
        encoded_images = [image_to_base64(image) for image in images]
        
        return jsonify({
        "query": query,
        "answer": answer[0],
        "pages": encoded_images,  # List of base64 strings
        "top_pages": top_pages.tolist()
        }), 200
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "Error processing query", "error": str(e), "trace": traceback.format_exc()}), 500
    finally:
        torch.cuda.empty_cache()

@app.route('/query/image', methods=['POST'])
def handle_query_image():
    """
    Handles a POST request to the '/query/image' endpoint that contains a filename and a query.
    Retrieves the corresponding image data from the HDF5 file and passes it to the
    generate_answer function to generate a text response based on the query and the
    retrieved image.

    Args:
        filename (str): The filename of the PDF document to retrieve images from.
        query (str): The query string to generate an answer for.

    Returns:
        A JSON response with a status message of "OK" and the following keys:

        * query (str): The query string.
        * answer (str): The generated answer.
        * pages (list[str]): A list of base64 strings, each representing an image retrieved from the PDF.
        * top_pages (list[int]): The indices of the top-scoring pages in the PDF document.
    """
    try:
        filename = request.form["filename"]
        query = request.form["query"]
        image = request.files['image']
        pil_image = Image.open(image)

        # Retrieve with adapter switching
        image_embedding = process_image([pil_image])
        
        pdf_embedding = get_embeddings_by_filename(filename)
        
        # Compute scores between the query embedding and the image embeddings
        scores = processor_retrieval.score_multi_vector(
            torch.from_numpy(image_embedding), 
            torch.from_numpy(pdf_embedding)
        )
        
        # Get the index of the top-scoring page
        top_pages = scores.numpy()[0].argsort()[-RETRIEVE_K:][::-1]
        images = [get_image_by_filename(filename, page) for page in top_pages]
        
        # Generate an answer based on the query and the top-scoring page
        answer = generate_answer(query, images)
        
        encoded_images = [image_to_base64(image) for image in images]
        
        return jsonify({
        "query": query,
        "answer": answer[0],
        "pages": encoded_images,  # List of base64 strings
        "top_pages": top_pages.tolist()
        }), 200
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"status": "Error processing query", "error": str(e), "trace": traceback.format_exc()}), 500
    finally:
        torch.cuda.empty_cache()
        
@app.route('/filenames', methods=['GET'])
def get_filenames():
    """
    Handles a GET request to the '/filenames' endpoint that returns a list of filenames
    present in the knowledge base HDF5 file.
    """
    with h5py.File(KNOWLEDGE_BASE, 'r') as f:
        filenames = list(f.keys())
    return jsonify({"filenames": filenames})

def image_to_base64(image: Image.Image, image_format: str = 'JPEG') -> str:
    buffered = BytesIO()
    image.save(buffered, format=image_format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str
    
def ndarray_to_base64(ndarr: np.ndarray, image_format: str = 'png') -> str:
    """
    Convert a NumPy ndarray (image) to a base64 encoded string.
    
    Args:
        ndarr (np.ndarray): The input image as a NumPy array.
        image_format (str): Format to encode the image (default is 'png').
        
    Returns:
        str: The base64 encoded string of the image.
    """
    # Encode the numpy array into the specified image format
    success, buffer = cv2.imencode(f'.{image_format}', ndarr)
    if not success:
        raise ValueError("Failed to encode image")
    
    # Convert buffer to bytes and then encode to base64 string
    img_bytes = buffer.tobytes()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)