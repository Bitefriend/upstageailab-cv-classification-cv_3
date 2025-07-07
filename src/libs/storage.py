import re
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from src import config
from src.util.helper import add_timestamp_prefix
from src.util.log import get_logger


@dataclass
class CloudStorageAuth:
    access_key: str
    secret_key: str
    region: str
    bucket: str
    endpoint: str


class AugmentedImageManager:
    """증강 이미지의 클라우드 스토리지 업로드/다운로드를 관리하는 클래스"""

    STATUS_OK = 200
    STATUS_NO_CONTENT = 204

    # 증강 이미지 압축파일 패턴: {timestamp}_{augment_tag}.tar.gz
    ARCHIVE_PATTERN = r"^(\d{14})_([a-zA-Z0-9_]+)\.tar\.gz$"

    # S3 디렉토리 구조
    PREFIX_ROOT_DIR = "augmented_images"

    UNIT_MB = 1024 * 1024

    def __init__(self, auth: CloudStorageAuth, root_dir: str | None = None):
        self.bucket = auth.bucket
        self.client = boto3.client(
            "s3",
            aws_access_key_id=auth.access_key,
            aws_secret_access_key=auth.secret_key,
            region_name=auth.region,
            endpoint_url=auth.endpoint,
            config=Config(signature_version="s3v4"),
        )
        self._logger = get_logger("augmented_image_manager")
        self.root_dir = f"{self.PREFIX_ROOT_DIR}/{root_dir}" if root_dir else self.PREFIX_ROOT_DIR

    def upload(self, image_directory: Path, augment_tag: str = "augmented", experiment_name: str | None = None) -> str:
        """증강된 이미지 디렉토리를 압축하여 클라우드에 업로드"""
        # 디렉토리 존재 확인
        if not image_directory.exists() or not image_directory.is_dir():
            raise ValueError(f"이미지 디렉토리를 다시 확인해주세요: {image_directory}")

        # 압축 파일명 생성
        archive_name = add_timestamp_prefix(f"{augment_tag}.tar.gz")

        # 임시 디렉토리에서 압축 작업 수행
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / archive_name

            # 이미지 디렉토리 압축
            self._logger.info(f"Compressing {image_directory}...")
            self._compress_directory(image_directory, archive_path)

            # S3 key 생성
            s3_key = f"{self.root_dir}/{experiment_name}/" if experiment_name else f"{self.root_dir}/" + archive_name

            # S3에 업로드
            self._logger.info(f"Uploading to {s3_key}...")
            if self._upload_file(archive_path, s3_key):
                self._logger.info(f"Successfully uploaded {archive_path.stat().st_size:,} bytes")
                return s3_key
            raise RuntimeError(f"Failed to upload {archive_name}")

    def download(self, s3_key: str, target_directory: Path) -> str:
        """클라우드에서 증강 이미지를 다운로드하고 압축 해제"""

        # 타겟 디렉토리가 없다면 생성
        target_directory.mkdir(parents=True, exist_ok=True)

        # 임시 파일로 다운로드 후 압축 해제
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            self._logger.info(f"Downloading {s3_key}...")
            if not self._download_file(s3_key, temp_path):
                raise RuntimeError(f"Failed to download {s3_key}")

            # 압축 해제
            self._logger.info(f"Extracting to {target_directory}...")

            s3_path = Path(s3_key)
            archive_stem = s3_path.stem  # .tar.gz에서 .gz 제거 -> 예시) "20250107_rotation.tar"
            if archive_stem.endswith(".tar"):
                archive_stem = Path(archive_stem).stem  # .tar 제거 -> 예시)  "20250107_rotation"

            extracted_dir = target_directory / archive_stem
            extracted_dir.mkdir(parents=True, exist_ok=True)

            self._extract_archive(temp_path, extracted_dir)

            self._logger.info(f"Successfully extracted to {extracted_dir}")
            return extracted_dir
        finally:
            if temp_path.exists():  # 임시 파일 삭제
                temp_path.unlink()

    def list_archives(self, experiment_name: str | None = None) -> list[dict]:
        """업로드된 증강 이미지 압축 파일 목록 조회"""
        # S3 prefix 설정
        prefix = f"{self.root_dir}/{experiment_name}/" if experiment_name else f"{self.root_dir}/"

        archives = []
        try:
            self._logger.info(f"Listing archives in {prefix}")
            response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)

            if "Contents" not in response:
                self._logger.info("No archives found")
                return archives

            for obj in response["Contents"]:
                key = obj["Key"]
                filename = Path(key).name

                # 패턴 매칭으로 유효한 압축 파일 필터링
                match = re.match(self.ARCHIVE_PATTERN, filename)
                if match:
                    timestamp, augment_tag = match.groups()

                    # 타임스탬프를 datetime으로 변환
                    upload_time = datetime.strptime(timestamp, "%Y%m%d%H%M%S")

                    archives.append(
                        {
                            "key": key,
                            "filename": filename,
                            "augment_tag": augment_tag,
                            "upload_time": upload_time,
                            "size_bytes": obj["Size"],
                            "size_mb": round(obj["Size"] / self.UNIT_MB, 2),
                        }
                    )

            # 최신 순으로 정렬
            archives.sort(key=lambda x: x["upload_time"], reverse=True)

            self._logger.info(f"Found {len(archives)} archives")
            return archives

        except ClientError as e:
            self._logger.error(f"Failed to list archives: {e}")
            return []

    def delete(self, s3_key: str) -> bool:
        """증강 이미지 압축 파일 삭제"""
        try:
            self._logger.info(f"Deleting {s3_key}")
            response = self.client.delete_object(Bucket=self.bucket, Key=s3_key)
            return self._check_response(response, "delete", s3_key)
        except ClientError as e:
            self._logger.error(f"Failed to delete {s3_key}: {e}")
            return False

    # Private methods
    def _upload_file(self, local_path: Path, s3_key: str) -> bool:
        """파일을 S3에 업로드"""
        try:
            with local_path.open("rb") as f:
                response = self.client.put_object(Bucket=self.bucket, Key=s3_key, Body=f)
            return self._check_response(response, "upload", s3_key)
        except (OSError, ClientError) as e:
            self._logger.error(f"Upload failed: {e}")
            return False

    def _download_file(self, s3_key: str, local_path: Path) -> bool:
        """S3에서 파일 다운로드"""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=s3_key)
            with local_path.open("wb") as f:
                f.write(response["Body"].read())
            return True
        except (OSError, ClientError) as e:
            self._logger.error(f"Download failed: {e}")
            return False

    def _check_response(self, response: dict, operation: str, key: str) -> bool:
        """API 응답 상태 확인"""
        status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        success = status_code in [self.STATUS_OK, self.STATUS_NO_CONTENT]

        if not success:
            self._logger.warning(f"{operation} failed for {key}: status {status_code}")

        return success

    @staticmethod
    def _compress_directory(source_dir: Path, output_path: Path):
        """디렉토리를 tar.gz로 압축"""
        with tarfile.open(output_path, "w:gz") as tar:
            # 디렉토리명을 아카이브의 루트로 사용
            for file in source_dir.rglob("*"):
                tar.add(file, arcname=file.name)

    @staticmethod
    def _extract_archive(archive_path: Path, extract_dir: Path):
        """tar.gz 파일 압축 해제"""
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

    @classmethod
    def create(cls, root_directory: str | None = None) -> Self:
        """Factory method로 인스턴스 생성"""
        return cls(
            auth=CloudStorageAuth(
                access_key=config.NCLOUD_ACCESS_KEY,
                secret_key=config.NCLOUD_SECRET_KEY,
                region=config.NCLOUD_STORAGE_REGION,
                bucket=config.NCLOUD_STORAGE_BUCKET,
                endpoint=config.NCLOUD_STORAGE_ENDPOINT_URL,
            ),
            root_dir=root_directory,
        )
