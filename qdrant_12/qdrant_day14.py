import os

from dotenv import load_dotenv

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Embedding model
from sentence_transformers import SentenceTransformer

# LLM
from groq import Groq


# ============================================================
# PART 1 — LOAD API KEYS FROM .env
# ============================================================

# Load variables from the .env file
load_dotenv()

# Get Qdrant credentials
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Get Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# PART 2 — CONNECT TO QDRANT
# ============================================================

# Create a connection with Qdrant Cloud
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant Cloud!")


# ============================================================
# PART 3 — CREATE QDRANT COLLECTION
# ============================================================

# Name of our collection in Qdrant
COLLECTION_NAME = "knowledge"

# all-MiniLM-L6-v2 creates vectors containing 384 numbers
EMBEDDING_SIZE = 384


# If the collection already exists,
# delete it so that we can start fresh
if client.collection_exists(COLLECTION_NAME):

    print(f"Deleting existing collection: {COLLECTION_NAME}")

    client.delete_collection(COLLECTION_NAME)


# Create a new collection
client.create_collection(
    collection_name=COLLECTION_NAME,

    vectors_config=VectorParams(

        # Every vector stored in this collection
        # will contain 384 numbers
        size=EMBEDDING_SIZE,

        # Qdrant will use cosine distance
        # to find similar vectors
        distance=Distance.COSINE,
    ),
)

print(f"Created collection: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")


# ============================================================
# PART 4 — LOAD OUR KNOWLEDGE
# ============================================================

# Open knowledge.txt
with open("knowledge.txt", "r", encoding="utf-8") as f:

    # Read the file line by line
    #
    # line.strip() removes unnecessary spaces/newlines
    #
    # if line.strip() ignores empty lines
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]


# Example:
#
# documents = [
#     "Employees receive 24 days of paid leave per year.",
#     "Employees work from the office on Tuesday...",
#     "Employees receive Rs 3000 per month for gym reimbursement.",
#     ...
# ]

print(f"Loaded {len(documents)} documents")


# ============================================================
# PART 5 — CREATE EMBEDDINGS
# ============================================================

print("Loading embedding model...")


# Load the embedding model
#
# This model converts text into vectors
# Each vector contains 384 numbers
model = SentenceTransformer("all-MiniLM-L6-v2")


print("Embedding model ready!")


# Convert every document into an embedding
#
# documents:
# ["Leave information", "Office information", ...]
#
# becomes:
#
# embeddings:
# [
#     [384 numbers],
#     [384 numbers],
#     ...
# ]
embeddings = model.encode(documents)


print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")


# ============================================================
# PART 6 — CREATE QDRANT POINTS
# ============================================================

# Empty list where we will store all our Qdrant points
points = []


# Go through every embedding
#
# i = index
# embedding = current embedding
#
# Example:
#
# i = 0 → embedding of document 0
# i = 1 → embedding of document 1
# i = 2 → embedding of document 2
#
for i, embedding in enumerate(embeddings):


    # Create ONE Qdrant point
    #
    # A point contains:
    #
    # ID
    # Vector
    # Payload
    point = PointStruct(

        # Give every point a unique ID
        #
        # i starts from 0, so we use i + 1
        #
        # i = 0 → ID 1
        # i = 1 → ID 2
        # i = 2 → ID 3
        id=i + 1,


        # Store the embedding/vector
        #
        # .tolist() converts the NumPy array
        # into a normal Python list
        vector=embedding.tolist(),


        # Store the original text along with the vector
        #
        # This is called the PAYLOAD
        payload={
            "text": documents[i]
        }
    )


    # Add this point to our points list
    #
    # After the loop finishes:
    #
    # points = [
    #     Point 1,
    #     Point 2,
    #     Point 3,
    #     ...
    # ]
    points.append(point)


# ============================================================
# PART 7 — UPLOAD POINTS TO QDRANT
# ============================================================

# Upload all our points to Qdrant
#
# upsert means:
# insert if new
# update if the same ID already exists
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Uploaded {len(points)} documents to Qdrant!")


# ============================================================
# PART 8 — SEARCH QDRANT
# ============================================================

def search(query, top_k=3):

    # --------------------------------------------------------
    # STEP 1 — Convert the question into an embedding
    # --------------------------------------------------------

    # Example:
    #
    # "How many vacation days do I get?"
    #
    # becomes:
    #
    # [0.12, -0.43, 0.21, ...]
    #
    # 384 numbers
    query_vector = model.encode(query).tolist()


    # --------------------------------------------------------
    # STEP 2 — Search Qdrant
    # --------------------------------------------------------

    # Ask Qdrant:
    #
    # "Find the most similar vectors to my query_vector."
    #
    # limit=3 means:
    # Give me the TOP 3 similar results.
    #
    # with_payload=True means:
    # Also give me the original text stored with each vector.
    results = client.query_points(
        collection_name=COLLECTION_NAME,

        # The question's embedding
        query=query_vector,

        # How many results we want
        limit=top_k,

        # Include the stored text
        with_payload=True,

    ).points


    # Return the results
    #
    # If top_k = 3:
    #
    # results = [
    #     result 1,
    #     result 2,
    #     result 3
    # ]
    return results


# ============================================================
# PART 9 — TEST SEARCH
# ============================================================

# User's question
query = "How many vacation days do I get?"


# Search Qdrant
#
# We want the top 3 most similar documents
results = search(query, top_k=3)


print("\nSearch results:")


# results contains up to 3 results
#
# So this loop runs up to 3 times
for result in results:

    # Similarity score calculated by Qdrant
    print(f"Score: {result.score:.3f}")

    # Get the original text from the payload
    print(result.payload["text"])

    print()


# ============================================================
# PART 10 — CONNECT TO GROQ
# ============================================================

# Create Groq client
groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# PART 11 — ASK THE LLM
# ============================================================

def ask_llm(question, context):

    # Create a prompt for the LLM
    #
    # The LLM will receive:
    #
    # 1. Retrieved context
    # 2. User's question
    #
    # It should answer only using the context.
    prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided information."
"""


    # Send the prompt to Llama through Groq
    response = groq_client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # Extract and return the actual answer
    return response.choices[0].message.content


# ============================================================
# PART 12 — COMPLETE RAG PIPELINE
# ============================================================

# The user's question
question = "How many vacation days do I get?"


# ------------------------------------------------------------
# STEP 1 — RETRIEVAL
# ------------------------------------------------------------

# Search Qdrant for the top 3 relevant documents
results = search(question, top_k=3)


# ------------------------------------------------------------
# STEP 2 — CREATE CONTEXT
# ------------------------------------------------------------

# Extract the original text from each result
#
# Suppose results contains:
#
# Result 1 → "Employees receive 24 days..."
# Result 2 → "Employees have a 90 day notice period."
# Result 3 → "Employees work from the office..."
#
# They are combined into ONE string.
context = "\n".join(
    result.payload["text"]
    for result in results
)


# Now context looks like:
#
# Employees receive 24 days of paid leave per year.
# Employees have a 90 day notice period.
# Employees work from the office...


# ------------------------------------------------------------
# STEP 3 — GENERATION
# ------------------------------------------------------------

# Send:
#
# question + retrieved context
#
# to the LLM
answer = ask_llm(question, context)


# ------------------------------------------------------------
# STEP 4 — PRINT FINAL ANSWER
# ------------------------------------------------------------

print("\nFinal Answer:")
print(answer)