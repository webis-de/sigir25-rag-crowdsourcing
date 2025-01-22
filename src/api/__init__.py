import logging
import yaml
import time
import threading
import os
from functools import lru_cache

from fastapi import FastAPI, Request, Depends, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Annotated, TypeAlias, Dict, Any
import datetime
import json
import uuid


class Config:
    def __init__(self, config_path: str, reload_interval: int = 30, delimiter: str = "."):
        """
        Initialize the config

        Args:
            config_path: Path to the YAML config file
            reload_interval: How often to check for config changes (in seconds)
            delimiter: Character to use for separating nested keys (e.g., "database.url")
        """
        self.config_path = config_path
        self.reload_interval = reload_interval
        self.delimiter = delimiter
        self.last_modified_time = 0
        self._config: dict[str, Any] = {}
        self._load_config()

        # Start background thread for config reloading
        self.thread = threading.Thread(target=self._reload_config_loop, daemon=True)
        self.thread.start()

    def _load_config(self) -> None:
        """Load the config file and update last modified time"""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            self.last_modified_time = os.path.getmtime(self.config_path)
        except Exception as e:
            print(f"Error loading config: {e}")
            # Keep the old config if loading fails

    def _has_config_changed(self) -> bool:
        """Check if the config file has been modified"""
        try:
            current_mtime = os.path.getmtime(self.config_path)
            return current_mtime > self.last_modified_time
        except Exception:
            return False

    def _reload_config_loop(self) -> None:
        """Background thread that periodically checks for config changes"""
        while True:
            if self._has_config_changed():
                self._load_config()
            time.sleep(self.reload_interval)

    def _get_nested(self, keys: list[str], config: dict[str, Any], default: Any) -> Any:
        """Recursively get nested dictionary values"""
        if not keys:
            return config

        if not isinstance(config, dict):
            return default

        key = keys[0]
        if key not in config:
            return default

        if len(keys) == 1:
            return config[key]

        return self._get_nested(keys[1:], config[key], default)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the config, returning a default if not found.
        Supports nested keys using the delimiter (e.g., "database/url").

        Args:
            key: The configuration key to look up (can be nested, e.g., "database/url")
            default: The default value to return if the key is not found

        Returns:
            The value from the configuration, or the default if not found
        """
        if self.delimiter in key:
            keys = key.split(self.delimiter)
            return self._get_nested(keys, self._config, default)
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to config values"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator usage"""
        return self.get(key) is not None


@lru_cache()
def get_config(config_path: str = "config.yaml", reload_interval: int = 30, delimiter: str = ".") -> Config:
    return Config(config_path, reload_interval, delimiter)


ConfigDependency: TypeAlias = Annotated[Config, Depends(get_config)]


class FormResponse(SQLModel, table=True):
    # Primary ID
    id: str = Field(primary_key=True)
    # The UUID of the questionnaire form this was submitted from
    submission_id: str
    # The timestamp this submission was received at
    submitted_at: datetime.datetime
    # Prolific user ID
    prolific_pid: str
    # Prolific study ID
    prolific_study_id: str | None = None
    # Prolific session ID
    prolific_session_id: str | None = None
    # Prolific consent check
    consent: bool
    # Stores copy of the submitted form data as JSON-encoded str
    data: str

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


def _session(config: ConfigDependency) -> Session:
    sqlite_file_name = config.get("db.path", "responses.db")
    sqlite_url = f"sqlite:///{sqlite_file_name}"
    engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


SessionDependency = Annotated[Session, Depends(_session)]

logger = logging.getLogger('uvicorn.error')

# FastAPI app
app = FastAPI(title="Form Submission API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_json(data_str: str) -> Dict:
    """Parse JSON string into a dictionary"""
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return {}


@app.post("/submit", status_code=200)
async def submit_form(
    request: Request,
    response: Response,
    config: ConfigDependency,
    session: SessionDependency,
):
    """Handle form submission"""
    try:
        form_data = await request.form()
        form_data = dict(form_data)
        prolific_info = parse_json(form_data.get("prolific_info", '{}'))
        submission_id = str(form_data["submission_id"])
        study_name = str(form_data["study_name"])
        consent = prolific_info.get("consent", False)
        # Create response record
        record = FormResponse(
            id=str(uuid.uuid4()),
            submission_id=submission_id,
            submitted_at=datetime.datetime.now(datetime.UTC),
            prolific_pid=prolific_info.get('prolific_pid'),
            prolific_study_id=prolific_info.get('prolific_study_id'),
            prolific_session_id=prolific_info.get('prolific_session_id'),
            consent=consent,
            data=json.dumps(form_data)
        )
        session.add(record)
        session.commit()

        if consent:
            return {
                "completionCode": config.get(f"codes.{study_name}.done", "ERROR")
            }
        else:
            return {
                "completionCode": config.get(f"codes.{study_name}.no_consent", "ERROR")
            }

    except Exception as e:
        logger.exception(e)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {
            "message": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
