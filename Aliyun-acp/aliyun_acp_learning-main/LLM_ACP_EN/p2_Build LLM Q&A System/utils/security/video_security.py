#encoding:utf-8
# python version >= 3.6
import time

from alibabacloud_green20220302.client import Client
from alibabacloud_green20220302 import models
from alibabacloud_tea_openapi.models import Config
import json
import os

access_key_id = os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID']
access_key_secret = os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']


config = Config(
            # The AccessKey of your Alibaba Cloud account has access to all APIs. It is recommended to use a RAM user for API access or daily operations.
            # Strongly recommend not saving the AccessKey ID and AccessKey Secret in the project code, otherwise it may lead to AccessKey leakage, threatening the security of all resources under your account.
            # Common ways to obtain environment variables:
            # Get RAM user AccessKey ID: os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID']
            # Get RAM user AccessKey Secret: os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            # Connection timeout, in milliseconds (ms).
            connect_timeout=3000,
            # Read timeout, in milliseconds (ms).
            read_timeout=6000,
            # The region and endpoint should be modified according to actual conditions.
            region_id='cn-shanghai',
            endpoint='green-cip.cn-shanghai.aliyuncs.com'
)

client = Client(config)


def submit_task(video_url):

    service_parameters = {
        'url': video_url
    }
    video_moderation_request = models.VideoModerationRequest(
        # Detection type: videoDetection
        service='videoDetection',
        service_parameters=json.dumps(service_parameters)
    )

    try:
        response = client.video_moderation(video_moderation_request)
        if response.status_code == 200:
            result = response.body
            print('Video submit task:{}'.format(result.data))
            return result.data.task_id
        else:
            print('Video submit task failed. Status:{} ,Result:{}'.format(response.status_code, response))
    except Exception as err:
        print(err)


def get_result(task_id):
    # taskId returned when submitting the task.
    service_parameters = {
        "taskId": task_id
    }
    video_moderation_result_request = models.VideoModerationResultRequest(
        # Detection type: videoDetection
        service='videoDetection',
        service_parameters=json.dumps(service_parameters)
    )

    try:
        response = client.video_moderation_result(video_moderation_result_request)
        if response.status_code == 200:
            result = response.body
            print('Video detect result:{}'.format(result))
        else:
            print('Video detect failed. Status:{} ,Result:{}'.format(response.status_code, response))
    except Exception as err:
        print(err)


def detect(video_url):
    task_id = submit_task(video_url)
    time.sleep(3)
    get_result(task_id)


if __name__ == "__main__":
    video_url = ''
    # task_id = submit_task(video_url)
    task_id = 'vi_f_YLZysINUYOMP5fIHhRCIrL-1AzLt6'
    get_result(task_id)