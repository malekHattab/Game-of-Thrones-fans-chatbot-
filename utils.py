from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Scene Tags
scene_tags = {
    "The Battle of the Blackwater Bay": ["battle", "Blackwater Bay", "Stannis Baratheon", "King's Landing", "Tyrion Lannister", "wildfire", "siege"],
    "The Battle of the Red Keep": ["battle", "Red Keep", "King's Landing", "Stannis Baratheon", "Cersei Lannister", "Tyrion Lannister", "wildfire"],
    "The Battle of the Bastards": ["battle", "Winterfell", "Ramsay Bolton", "Jon Snow", "Sansa Stark", "Brienne of Tarth", "Podrick Payne", "Tormund Giantsbane"],
    "The Battle of the Gold Cloaks": ["battle", "King's Landing", "Gold Cloaks", "Stannis Baratheon", "Cersei Lannister", "Tyrion Lannister", "wildfire"],
    "The Battle of the Iron Islands": ["battle", "Iron Islands", "Theon Greyjoy", "Euron Greyjoy", "Yara Greyjoy", "Arya Stark", "Theon's fate"],
    "The Battle of the Riverlands": ["battle", "Riverlands", "Tully lands", "Lannister army", "Stark army", "Tyrion Lannister", "Cersei Lannister", "Robb Stark"],
    "The Battle of the North": ["battle", "North", "Stark lands", "White Walkers", "Night King", "Jon Snow", "Daenerys Targaryen", "Arya Stark", "Sansa Stark"],
    "The Battle of King's Landing": ["battle", "King's Landing", "Daenerys Targaryen", "Jon Snow", "Cersei Lannister", "Tyrion Lannister", "wildfire", "dragonfire"],
    "The Battle of Winterfell": ["battle", "Winterfell", "White Walkers", "Night King", "Jon Snow", "Daenerys Targaryen", "Arya Stark", "Sansa Stark", "Brienne of Tarth"],
    "The Battle of the Narrow Sea": ["battle", "Narrow Sea", "Daenerys Targaryen", "Khal Drogo", "Viserys Targaryen", "Dothraki", "Targaryen army"],
    "The Battle of the Trident": ["battle", "Trident", "Robb Stark", "Stannis Baratheon", "Renly Baratheon", "Tyrion Lannister", "Cersei Lannister"]
}

# Character Tags
characters = {
    "Arya Stark": ["Arya", "Stark", "Faceless Assassin", "No One", "Water Dancer", "Needle", "wolf", "free spirit", "brave", "loyal", "skilled fighter"],
    "Sansa Stark": ["Sansa", "Stark", "Lady of Winterfell", "Queen in the North", "Bride of Ramsay", "Bride of Tyrion", "strong", "resilient", "beautiful", "cunning", "manipulative"],
    "Brienne of Tarth": ["Brienne", "Tarth", "Kingsguard", "Lady of Tarth", "Oathkeeper", "honorable", "loyal", "skilled fighter", "brave", "powerful", "beautiful"],
    "Tyrion Lannister": ["Tyrion", "Lannister", "Imp", "dwarf", "witty", "intelligent", "alcoholic", "poet", "Hand of the King", "brave", "loyal", "powerful", "cunning"],
    "Daenerys Targaryen": ["Daenerys", "Targaryen", "Mother of Dragons", "Breaker of Chains", "Queen of the Andals and the First Men", "Khaleesi", "dragon queen", "conqueror", "brave", "loyal", "powerful", "beautiful"],
    "Cersei Lannister": ["Cersei", "Lannister", "Queen Regent", "Queen of the Seven Kingdoms", "manipulative", "power-hungry", "cunning", "beautiful", "brave", "loyal", "powerful"],
    "Jon Snow": ["Jon", "Snow", "Lord Commander", "Night's Watch", "King in the North", "Targaryen", "trueborn", "bastard", "leader", "brave", "loyal", "powerful", "beautiful"],
    "Ramsay Bolton": ["Ramsay", "Bolton", "Flayed Man", "Lord of Winterfell", "cruel", "sadistic", "power-hungry", "tyrant", "brave", "loyal", "powerful", "beautiful"],
    "The Night King": ["Night King", "White Walker", "King of the Dead", "undead", "immortal", "powerful", "cold", "calculating", "brave", "loyal", "powerful"],
    "The Mountain": ["The Mountain", "Gregor Clegane", "Kingslayer", "Ser", "brutal", "powerful", "loyal", "fearsome", "brave", "beautiful"],
    "The Hound": ["The Hound", "Sandor Clegane", "Brother", "Kinslayer", "Ser", "brooding", "intimidating", "complex", "redemptive", "brave", "loyal", "powerful"],
    "Theon Greyjoy": ["Theon", "Greyjoy", "Reek", "Iron Islands", "pirate", "traitor", "tortured", "broken", "rebellious", "brave", "loyal", "powerful"],
    "Catelyn Stark": ["Catelyn", "Stark", "Lady of Winterfell", "Queen in the North", "motherly", "loyal", "strong", "resilient", "brave", "beautiful", "cunning"],
    "Robb Stark": ["Robb", "Stark", "King in the North", "King of the Trident", "brave", "honorable", "just", "young", "loyal", "powerful", "beautiful"],
    "Edmure Tully": ["Edmure", "Tully", "Riverlands", "Lord of Riverrun", "brotherly", "loyal", "brave", "impulsive", "brave", "loyal", "powerful"],
    "Margaery Tyrell": ["Margaery", "Tyrell", "Queen of the Seven Kingdoms", "Queen of King's Landing", "beautiful", "charismatic", "manipulative", "power-hungry", "brave", "loyal", "powerful"],
    "Oberyn Martell": ["Oberyn", "Martell", "Red Viper", "Prince of Dorne", "vengeful", "passionate", "charismatic", "skilled fighter", "brave", "loyal", "powerful"],
    "Tormund Giantsbane": ["Tormund", "Giantsbane", "Free Folk", "Wildling", "leader", "brave", "loyal", "witty", "tall", "powerful", "beautiful"],
    "Podrick Payne": ["Podrick", "Payne", "squire", "knight", "loyal", "brave", "funny", "awkward", "brave", "loyal", "powerful"],
    "Bran Stark": ["Bran", "Stark", "Three-Eyed Raven", "Greenseer", "prophetic", "wise", "mysterious", "telepathic", "brave", "loyal", "powerful", "beautiful"],
    "Samwell Tarly": ["Samwell", "Tarly", "Night's Watch", "Maester", "scholarly", "brave", "loyal", "fat", "brave", "loyal", "powerful"],
    "Gilly": ["Gilly", "wildling", "motherly", "strong", "resilient", "brave", "loyal", "powerful", "beautiful", "cunning"],
    "Melisandre": ["Melisandre", "Red Priestess", "Lord of Light", "R'hllor", "beautiful", "charismatic", "manipulative", "power-hungry", "brave", "loyal", "powerful"],
    "Stannis Baratheon": ["Stannis", "Baratheon", "King of the Seven Kingdoms", "King of the Iron Throne", "brotherly", "loyal", "brave", "just", "powerful", "beautiful"],
    "Shireen Baratheon": ["Shireen", "Baratheon", "Lady of Dragonstone", "daughterly", "beautiful", "weak", "vulnerable", "cursed", "brave", "loyal", "powerful"],
    "Arya's Direwolf": ["Nymeria", "direwolf", "Arya's wolf", "wild", "free", "loyal", "brave", "protective", "beautiful", "powerful"],
    "Jon's Direwolf": ["Ghost", "direwolf", "Jon's wolf", "loyal", "silent", "protective", "brave", "beautiful", "powerful"]
}


# Load PDF and split into sections
def load_and_split_pdf(pdf_path):
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_documents = text_splitter.split_documents(documents)
    return split_documents

# Tagging function
def tag_scenes_and_characters(documents):
    for doc in documents:
        doc.metadata["tags"] = []
        doc_text = getattr(doc, 'page_content', "")
        
        # Tag scenes
        for scene, keywords in scene_tags.items():
            if any(keyword.lower() in doc_text.lower() for keyword in keywords):
                doc.metadata["tags"].append(scene)
        
        # Tag characters
        for character, keywords in characters.items():
            if any(keyword.lower() in doc_text.lower() for keyword in keywords):
                doc.metadata["tags"].append(character)
                
    return documents

# Q&A System Class
class GoTQASystem:
    def __init__(self, huggingface_token):
        self.huggingface_token = huggingface_token
        self.vector_store = None
        self.conversation_chain = None

    def create_embeddings(self, documents):
        """Create embeddings using HuggingFace's model and store in FAISS with metadata."""
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.vector_store = FAISS.from_documents(documents, embedding_model)
            print("✓ Embeddings created successfully!")
        except Exception as e:
            print(f"Error creating embeddings: {str(e)}")
            raise

    def setup_conversation_chain(self):
        """Set up a conversation chain with LLMChain and custom context support."""
        if self.huggingface_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.huggingface_token

        try:
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",
                model_kwargs={"temperature": 0.5, "max_length": 512}
            )
            # Additional code for conversation setup...
        except Exception as e:
            print(f"Error setting up conversation chain: {str(e)}")
            raise
