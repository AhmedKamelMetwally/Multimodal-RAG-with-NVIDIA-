from urllib.parse import urlparse

from pypdf import PdfReader

from langflow.base.data.base_file import BaseFileComponent
from langflow.inputs.inputs import BoolInput, DropdownInput, FloatInput, IntInput, MessageTextInput, SecretStrInput
from langflow.schema.data import Data
from langchain_openai import ChatOpenAI
from typing import Any

from langflow.base.compressors.model import LCCompressorComponent
from langflow.field_typing import BaseDocumentCompressor
from langflow.inputs.inputs import SecretStrInput
from langflow.io import DropdownInput, StrInput
from langflow.schema.dotdict import dotdict
from langflow.template.field.base import Output
from copy import deepcopy
from typing import TYPE_CHECKING

from chromadb.config import Settings
from langchain_chroma import Chroma
from typing_extensions import override

from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.base.vectorstores.utils import chroma_collection_to_data
from langflow.inputs.inputs import BoolInput, DropdownInput, HandleInput, IntInput, StrInput
from langflow.schema.data import Data
from typing import Any

from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings
from langflow.inputs.inputs import DropdownInput, SecretStrInput
from langflow.io import FloatInput, MessageTextInput
from langflow.schema.dotdict import dotdict

if TYPE_CHECKING:
    from langflow.schema.dataframe import DataFrame
class NvidiaIngestComponent(BaseFileComponent):
    display_name = "NVIDIA Retriever Extraction"
    description = "Multi-modal data extraction from documents using NVIDIA's NeMo API."
    documentation: str = "https://docs.nvidia.com/nemo/retriever/extraction/overview/"
    icon = "NVIDIA"
    beta = True

    try:
        from nv_ingest_client.util.file_processing.extract import EXTENSION_TO_DOCUMENT_TYPE

        # Supported file extensions from https://github.com/NVIDIA/nv-ingest/blob/main/README.md
        VALID_EXTENSIONS = ["pdf", "docx", "pptx", "jpeg", "png", "svg", "tiff", "txt"]
    except ImportError:
        msg = (
            "NVIDIA Retriever Extraction (nv-ingest) dependencies missing. "
            "Please install them using your package manager. (e.g. uv pip install langflow[nv-ingest])"
        )
        VALID_EXTENSIONS = [msg]

    inputs = [
        *BaseFileComponent._base_inputs,
        MessageTextInput(
            name="base_url",
            display_name="Base URL",
            info="The URL of the NVIDIA NeMo Retriever Extraction API.",
            required=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="NVIDIA API Key",
        ),
        BoolInput(
            name="extract_text",
            display_name="Extract Text",
            info="Extract text from documents",
            value=True,
        ),
        BoolInput(
            name="extract_charts",
            display_name="Extract Charts",
            info="Extract text from charts",
            value=False,
        ),
        BoolInput(
            name="extract_tables",
            display_name="Extract Tables",
            info="Extract text from tables",
            value=False,
        ),
        BoolInput(
            name="extract_images",
            display_name="Extract Images",
            info="Extract images from document",
            value=True,
        ),
        BoolInput(
            name="extract_infographics",
            display_name="Extract Infographics",
            info="Extract infographics from document",
            value=False,
            advanced=True,
        ),
        DropdownInput(
            name="text_depth",
            display_name="Text Depth",
            info=(
                "Level at which text is extracted (applies before splitting). "
                "Support for 'block', 'line', 'span' varies by document type."
            ),
            options=["document", "page", "block", "line", "span"],
            value="page",  # Default value
            advanced=True,
        ),
        BoolInput(
            name="split_text",
            display_name="Split Text",
            info="Split text into smaller chunks",
            value=True,
            advanced=True,
        ),
        IntInput(
            name="chunk_size",
            display_name="Chunk size",
            info="The number of tokens per chunk",
            value=500,
            advanced=True,
        ),
        IntInput(
            name="chunk_overlap",
            display_name="Chunk Overlap",
            info="Number of tokens to overlap from previous chunk",
            value=150,
            advanced=True,
        ),
        BoolInput(
            name="filter_images",
            display_name="Filter Images",
            info="Filter images (see advanced options for filtering criteria).",
            advanced=True,
            value=False,
        ),
        IntInput(
            name="min_image_size",
            display_name="Minimum Image Size Filter",
            info="Minimum image width/length in pixels",
            value=128,
            advanced=True,
        ),
        FloatInput(
            name="min_aspect_ratio",
            display_name="Minimum Aspect Ratio Filter",
            info="Minimum allowed aspect ratio (width / height). Images narrower than this will be filtered out.",
            value=0.2,
            advanced=True,
        ),
        FloatInput(
            name="max_aspect_ratio",
            display_name="Maximum Aspect Ratio Filter",
            info="Maximum allowed aspect ratio (width / height). Images taller than this will be filtered out.",
            value=5.0,
            advanced=True,
        ),
        BoolInput(
            name="dedup_images",
            display_name="Deduplicate Images",
            info="Filter duplicated images.",
            advanced=True,
            value=True,
        ),
        BoolInput(
            name="caption_images",
            display_name="Caption Images",
            info="Generate captions for images using the NVIDIA captioning model.",
            advanced=True,
            value=True,
        ),
        BoolInput(
            name="high_resolution",
            display_name="High Resolution (PDF only)",
            info=("Process pdf in high-resolution mode for better quality extraction from scanned pdf."),
            advanced=True,
            value=False,
        ),
    ]

    outputs = [
        *BaseFileComponent._base_outputs,
    ]

    def process_files(self, file_list: list[BaseFileComponent.BaseFile]) -> list[BaseFileComponent.BaseFile]:
        try:
            from nv_ingest_client.client import Ingestor
        except ImportError as e:
            msg = (
                "NVIDIA Retriever Extraction (nv-ingest) dependencies missing. "
                "Please install them using your package manager. (e.g. uv pip install langflow[nv-ingest])"
            )
            raise ImportError(msg) from e

        if not file_list:
            err_msg = "No files to process."
            self.log(err_msg)
            raise ValueError(err_msg)

        # Check if all files are PDFs when high resolution mode is enabled
        if self.high_resolution:
            for file in file_list:
                try:
                    with file.path.open("rb") as f:
                        PdfReader(f)
                except Exception as exc:
                    error_msg = "High-resolution mode only supports valid PDF files."
                    self.log(error_msg)
                    raise ValueError(error_msg) from exc

        file_paths = [str(file.path) for file in file_list]

        self.base_url: str | None = self.base_url.strip() if self.base_url else None
        if self.base_url:
            try:
                urlparse(self.base_url)
            except Exception as e:
                error_msg = f"Invalid Base URL format: {e}"
                self.log(error_msg)
                raise ValueError(error_msg) from e
        else:
            base_url_error = "Base URL is required"
            raise ValueError(base_url_error)

        self.log(
            f"Creating Ingestor for Base URL: {self.base_url!r}",
        )

        try:
            ingestor = (
                Ingestor(
                    message_client_kwargs={
                        "base_url": self.base_url,
                        "headers": {"Authorization": f"Bearer {self.api_key}"},
                        "max_retries": 3,
                        "timeout": 60,
                    }
                )
                .files(file_paths)
                .extract(
                    extract_text=self.extract_text,
                    extract_tables=self.extract_tables,
                    extract_charts=self.extract_charts,
                    extract_images=self.extract_images,
                    extract_infographics=self.extract_infographics,
                    text_depth=self.text_depth,
                    **({"extract_method": "nemoretriever_parse"} if self.high_resolution else {}),
                )
            )

            if self.extract_images:
                if self.dedup_images:
                    ingestor = ingestor.dedup(content_type="image", filter=True)

                if self.filter_images:
                    ingestor = ingestor.filter(
                        content_type="image",
                        min_size=self.min_image_size,
                        min_aspect_ratio=self.min_aspect_ratio,
                        max_aspect_ratio=self.max_aspect_ratio,
                        filter=True,
                    )

                if self.caption_images:
                    ingestor = ingestor.caption()

            if self.extract_text and self.split_text:
                ingestor = ingestor.split(
                    tokenizer="intfloat/e5-large-unsupervised",
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    params={"split_source_types": ["PDF"]},
                )

            result = ingestor.ingest()
        except Exception as e:
            ingest_error = f"Error during ingestion: {e}"
            self.log(ingest_error)
            raise

        self.log(f"Results: {result}")

        data: list[Data | None] = []
        document_type_text = "text"
        document_type_structured = "structured"

        # Result is a list of segments as determined by the text_depth option (if "document" then only one segment)
        # each segment is a list of elements (text, structured, image)
        for segment in result:
            if segment:
                for element in segment:
                    document_type = element.get("document_type")
                    metadata = element.get("metadata", {})
                    source_metadata = metadata.get("source_metadata", {})

                    if document_type == document_type_text:
                        data.append(
                            Data(
                                text=metadata.get("content", ""),
                                file_path=source_metadata.get("source_name", ""),
                                document_type=document_type,
                                metadata=metadata,
                            )
                        )
                    # Both charts and tables are returned as "structured" document type,
                    # with extracted text in "table_content"
                    elif document_type == document_type_structured:
                        table_metadata = metadata.get("table_metadata", {})

                        # reformat chart/table images as binary data
                        if "content" in metadata:
                            metadata["content"] = {"$binary": metadata["content"]}

                        data.append(
                            Data(
                                text=table_metadata.get("table_content", ""),
                                file_path=source_metadata.get("source_name", ""),
                                document_type=document_type,
                                metadata=metadata,
                            )
                        )
                    elif document_type == "image":
                        image_metadata = metadata.get("image_metadata", {})

                        # reformat images as binary data
                        if "content" in metadata:
                            metadata["content"] = {"$binary": metadata["content"]}

                        data.append(
                            Data(
                                text=image_metadata.get("caption", "No caption available"),
                                file_path=source_metadata.get("source_name", ""),
                                document_type=document_type,
                                metadata=metadata,
                            )
                        )
                    else:
                        self.log(f"Unsupported document type {document_type}")
        self.status = data or "No data"

        # merge processed data with BaseFile objects
        return self.rollup_data(file_list, data)
class NvidiaRerankComponent(LCCompressorComponent):
    display_name = "NVIDIA Rerank"
    description = "Rerank documents using the NVIDIA API."
    icon = "NVIDIA"

    inputs = [
        *LCCompressorComponent.inputs,
        SecretStrInput(
            name="api_key",
            display_name="NVIDIA API Key",
        ),
        StrInput(
            name="base_url",
            display_name="Base URL",
            value="https://integrate.api.nvidia.com/v1",
            refresh_button=True,
            info="The base URL of the NVIDIA API. Defaults to https://integrate.api.nvidia.com/v1.",
        ),
        DropdownInput(
            name="model",
            display_name="Model",
            options=["nv-rerank-qa-mistral-4b:1"],
            value="nv-rerank-qa-mistral-4b:1",
        ),
    ]

    outputs = [
        Output(
            display_name="Reranked Documents",
            name="reranked_documents",
            method="compress_documents",
        ),
    ]

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        if field_name == "base_url" and field_value:
            try:
                build_model = self.build_compressor()
                ids = [model.id for model in build_model.available_models]
                build_config["model"]["options"] = ids
                build_config["model"]["value"] = ids[0]
            except Exception as e:
                msg = f"Error getting model names: {e}"
                raise ValueError(msg) from e
        return build_config

    def build_compressor(self) -> BaseDocumentCompressor:
        try:
            from langchain_nvidia_ai_endpoints import NVIDIARerank
        except ImportError as e:
            msg = "Please install langchain-nvidia-ai-endpoints to use the NVIDIA model."
            raise ImportError(msg) from e
        return NVIDIARerank(api_key=self.api_key, model=self.model, base_url=self.base_url, top_n=self.top_n)
class ChromaVectorStoreComponent(LCVectorStoreComponent):
    """Chroma Vector Store with search capabilities."""

    display_name: str = "Chroma DB"
    description: str = "Chroma Vector Store with search capabilities"
    name = "Chroma"
    icon = "Chroma"

    inputs = [
        StrInput(
            name="collection_name",
            display_name="Collection Name",
            value="langflow",
        ),
        StrInput(
            name="persist_directory",
            display_name="Persist Directory",
        ),
        *LCVectorStoreComponent.inputs,
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"]),
        StrInput(
            name="chroma_server_cors_allow_origins",
            display_name="Server CORS Allow Origins",
            advanced=True,
        ),
        StrInput(
            name="chroma_server_host",
            display_name="Server Host",
            advanced=True,
        ),
        IntInput(
            name="chroma_server_http_port",
            display_name="Server HTTP Port",
            advanced=True,
        ),
        IntInput(
            name="chroma_server_grpc_port",
            display_name="Server gRPC Port",
            advanced=True,
        ),
        BoolInput(
            name="chroma_server_ssl_enabled",
            display_name="Server SSL Enabled",
            advanced=True,
        ),
        BoolInput(
            name="allow_duplicates",
            display_name="Allow Duplicates",
            advanced=True,
            info="If false, will not add documents that are already in the Vector Store.",
        ),
        DropdownInput(
            name="search_type",
            display_name="Search Type",
            options=["Similarity", "MMR"],
            value="Similarity",
            advanced=True,
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            advanced=True,
            value=10,
        ),
        IntInput(
            name="limit",
            display_name="Limit",
            advanced=True,
            info="Limit the number of records to compare when Allow Duplicates is False.",
        ),
    ]

    @override
    @check_cached_vector_store
    def build_vector_store(self) -> Chroma:
        """Builds the Chroma object."""
        try:
            from chromadb import Client
            from langchain_chroma import Chroma
        except ImportError as e:
            msg = "Could not import Chroma integration package. Please install it with `pip install langchain-chroma`."
            raise ImportError(msg) from e
        # Chroma settings
        chroma_settings = None
        client = None
        if self.chroma_server_host:
            chroma_settings = Settings(
                chroma_server_cors_allow_origins=self.chroma_server_cors_allow_origins or [],
                chroma_server_host=self.chroma_server_host,
                chroma_server_http_port=self.chroma_server_http_port or None,
                chroma_server_grpc_port=self.chroma_server_grpc_port or None,
                chroma_server_ssl_enabled=self.chroma_server_ssl_enabled,
            )
            client = Client(settings=chroma_settings)

        # Check persist_directory and expand it if it is a relative path
        persist_directory = self.resolve_path(self.persist_directory) if self.persist_directory is not None else None

        chroma = Chroma(
            persist_directory=persist_directory,
            client=client,
            embedding_function=self.embedding,
            collection_name=self.collection_name,
        )

        self._add_documents_to_vector_store(chroma)
        self.status = chroma_collection_to_data(chroma.get(limit=self.limit))
        return chroma

    def _add_documents_to_vector_store(self, vector_store: "Chroma") -> None:
        """Adds documents to the Vector Store."""
        ingest_data: list | Data | DataFrame = self.ingest_data
        if not ingest_data:
            self.status = ""
            return

        # Convert DataFrame to Data if needed using parent's method
        ingest_data = self._prepare_ingest_data()

        stored_documents_without_id = []
        if self.allow_duplicates:
            stored_data = []
        else:
            stored_data = chroma_collection_to_data(vector_store.get(limit=self.limit))
            for value in deepcopy(stored_data):
                del value.id
                stored_documents_without_id.append(value)

        documents = []
        for _input in ingest_data or []:
            if isinstance(_input, Data):
                if _input not in stored_documents_without_id:
                    documents.append(_input.to_lc_document())
            else:
                msg = "Vector Store Inputs must be Data objects."
                raise TypeError(msg)

        if documents and self.embedding is not None:
            self.log(f"Adding {len(documents)} documents to the Vector Store.")
            vector_store.add_documents(documents)
        else:
            self.log("No documents to add to the Vector Store.")

class NVIDIAEmbeddingsComponent(LCEmbeddingsModel):
    display_name: str = "NVIDIA Embeddings"
    description: str = "Generate embeddings using NVIDIA models."
    icon = "NVIDIA"

    inputs = [
        DropdownInput(
            name="model",
            display_name="Model",
            options=[
                "nvidia/nv-embed-v1",
                "snowflake/arctic-embed-I",
            ],
            value="nvidia/nv-embed-v1",
            required=True,
        ),
        MessageTextInput(
            name="base_url",
            display_name="NVIDIA Base URL",
            refresh_button=True,
            value="https://integrate.api.nvidia.com/v1",
            required=True,
        ),
        SecretStrInput(
            name="nvidia_api_key",
            display_name="NVIDIA API Key",
            info="The NVIDIA API Key.",
            advanced=False,
            value="NVIDIA_API_KEY",
            required=True,
        ),
        FloatInput(
            name="temperature",
            display_name="Model Temperature",
            value=0.1,
            advanced=True,
        ),
    ]

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        if field_name == "base_url" and field_value:
            try:
                build_model = self.build_embeddings()
                ids = [model.id for model in build_model.available_models]
                build_config["model"]["options"] = ids
                build_config["model"]["value"] = ids[0]
            except Exception as e:
                msg = f"Error getting model names: {e}"
                raise ValueError(msg) from e
        return build_config

    def build_embeddings(self) -> Embeddings:
        try:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        except ImportError as e:
            msg = "Please install langchain-nvidia-ai-endpoints to use the Nvidia model."
            raise ImportError(msg) from e
        try:
            output = NVIDIAEmbeddings(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                nvidia_api_key=self.nvidia_api_key,
            )
        except Exception as e:
            msg = f"Could not connect to NVIDIA API. Error: {e}"
            raise ValueError(msg) from e
        return output
class LanguageModelComponent(LCModelComponent):
    display_name = "Language Model"
    description = "Runs a language model given a specified provider."
    documentation: str = "https://docs.langflow.org/components-models"
    icon = "brain-circuit"
    category = "models"
    priority = 0  # Set priority to 0 to make it appear first

    inputs = [
        DropdownInput(
            name="provider",
            display_name="Model Provider",
            options=["OpenAI", "Anthropic", "Google"],
            value="OpenAI",
            info="Select the model provider",
            real_time_refresh=True,
            options_metadata=[{"icon": "OpenAI"}, {"icon": "Anthropic"}, {"icon": "GoogleGenerativeAI"}],
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=OPENAI_CHAT_MODEL_NAMES + OPENAI_REASONING_MODEL_NAMES,
            value=OPENAI_CHAT_MODEL_NAMES[0],
            info="Select the model to use",
            real_time_refresh=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="OpenAI API Key",
            info="Model Provider API key",
            required=False,
            show=True,
            real_time_refresh=True,
        ),
        MessageInput(
            name="input_value",
            display_name="Input",
            info="The input text to send to the model",
        ),
        MultilineInput(
            name="system_message",
            display_name="System Message",
            info="A system message that helps set the behavior of the assistant",
            advanced=False,
        ),
        BoolInput(
            name="stream",
            display_name="Stream",
            info="Whether to stream the response",
            value=False,
            advanced=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            info="Controls randomness in responses",
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
    ]

    def build_model(self) -> LanguageModel:
        provider = self.provider
        model_name = self.model_name
        temperature = self.temperature
        stream = self.stream

        if provider == "OpenAI":
            if not self.api_key:
                msg = "OpenAI API key is required when using OpenAI provider"
                raise ValueError(msg)

            if model_name in OPENAI_REASONING_MODEL_NAMES:
                # reasoning models do not support temperature (yet)
                temperature = None

            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                streaming=stream,
                openai_api_key=self.api_key,
            )
        if provider == "Anthropic":
            if not self.api_key:
                msg = "Anthropic API key is required when using Anthropic provider"
                raise ValueError(msg)
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                anthropic_api_key=self.api_key,
            )
        if provider == "Google":
            if not self.api_key:
                msg = "Google API key is required when using Google provider"
                raise ValueError(msg)
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                google_api_key=self.api_key,
            )
        msg = f"Unknown provider: {provider}"
        raise ValueError(msg)

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None) -> dotdict:
        if field_name == "provider":
            if field_value == "OpenAI":
                build_config["model_name"]["options"] = OPENAI_CHAT_MODEL_NAMES + OPENAI_REASONING_MODEL_NAMES
                build_config["model_name"]["value"] = OPENAI_CHAT_MODEL_NAMES[0]
                build_config["api_key"]["display_name"] = "OpenAI API Key"
            elif field_value == "Anthropic":
                build_config["model_name"]["options"] = ANTHROPIC_MODELS
                build_config["model_name"]["value"] = ANTHROPIC_MODELS[0]
                build_config["api_key"]["display_name"] = "Anthropic API Key"
            elif field_value == "Google":
                build_config["model_name"]["options"] = GOOGLE_GENERATIVE_AI_MODELS
                build_config["model_name"]["value"] = GOOGLE_GENERATIVE_AI_MODELS[0]
                build_config["api_key"]["display_name"] = "Google API Key"
        elif field_name == "model_name" and field_value.startswith("o1") and self.provider == "OpenAI":
            # Hide system_message for o1 models - currently unsupported
            if "system_message" in build_config:
                build_config["system_message"]["show"] = False
        elif field_name == "model_name" and not field_value.startswith("o1") and "system_message" in build_config:
            build_config["system_message"]["show"] = True
        return build_config
