```python
import boto3
import io
from typing import Any
import pickle
import os


class S3Buckets:

    PICKLE_FORMAT = 'pkl'
    PLOT_FORMAT = 'plt'
    PLOT_FORMAT_AX = 'ax'
    DATA_FRAME_FORMAT = 'df'
    SHAP_FORMAT = 'shap'
    bucket_name: str
    s3_client: Any = None

    def __init__(
        self,
        _bucket_name: str = ""
    ) -> None:
        if _bucket_name:
            self.bucket_name = _bucket_name
        else:
            self.bucket_name = os.getenv("BUCKET_NAME")
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("USER_S3_MODEL"),
            aws_secret_access_key=os.getenv("USER_SECRET_MODEL")
        )

    def get_list_files_by_prefix(
        self,
        prefix_path: str
    ) -> str:
        print(
            f"bucket_name: {self.bucket_name} "
            f"prefix_path: {prefix_path}"
        )

        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix_path
        )
        l_pp = [obj["Key"] for obj in response.get("Contents", [])]
        return l_pp

    def get_file_by_key(
        self,
        key: str
    ) -> Any:
        response = None
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
        except Exception as e:
            print(f"error: {e}")
        return response

    def save_file(
        self,
        key: str,
        obj: Any,
        format: str
    ) -> None:
        if format == self.PICKLE_FORMAT:
            buffer = io.BytesIO()
            pickle.dump(obj, buffer)
            buffer.seek(0)
            self.s3_client.upload_fileobj(
                buffer,
                self.bucket_name,
                key
            )
        elif format == self.PLOT_FORMAT:
            buffer = io.BytesIO()
            obj.savefig(buffer, format='png')
            buffer.seek(0)
            obj.close()
            self.s3_client.upload_fileobj(
                buffer,
                self.bucket_name,
                key
            )
        elif format == self.PLOT_FORMAT_AX:
            buffer = io.BytesIO()
            obj.savefig(buffer, format='png')
            buffer.seek(0)
            obj.close()
            self.s3_client.upload_fileobj(
                buffer,
                self.bucket_name,
                key
            )
        elif format == self.DATA_FRAME_FORMAT:
            buffer = io.StringIO()
            obj.to_csv(buffer, index=False)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue()
            )
        elif format == self.SHAP_FORMAT:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=obj,
                ContentType='image/png'
            )
```
