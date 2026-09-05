# coding=utf-8
# python version >= 3.6
from alibabacloud_green20220302.client import Client
from alibabacloud_green20220302 import models
from alibabacloud_tea_openapi.models import Config
import json
import os

access_key_id = os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID']
access_key_secret = os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']

config = Config(
    # The AccessKey of your Alibaba Cloud account has access to all APIs. It is recommended to use a RAM user for API access or daily operations.
    # Strongly recommend not saving the AccessKey ID and AccessKey Secret in the project code, as it may lead to AccessKey leakage, threatening the security of all resources under your account.
    # Common ways to obtain environment variables:
    # Get RAM user AccessKey ID: os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID']
    # Get RAM user AccessKey Secret: os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
    access_key_id=access_key_id,
    access_key_secret=access_key_secret,
    # Connection timeout time in milliseconds (ms)
    connect_timeout=10000,
    # Read timeout time in milliseconds (ms)
    read_timeout=3000,
    region_id='cn-hangzhou',
    endpoint='green-cip.cn-hangzhou.aliyuncs.com'
)
client = Client(config)


def detect(text, model="llm_query_moderation"):
    service_parameters = {
        'content': text
    }
    text_moderation_plusRequest = models.TextModerationPlusRequest(
        # Detection type
        service=model,
        service_parameters=json.dumps(service_parameters)
    )

    try:
        response = client.text_moderation_plus(text_moderation_plusRequest)
        if response.status_code == 200:
            # Call succeeded
            result = response.body
            # print('response success. result:{}'.format(result))
            if result.code == 200:
                result_data = result.data
                print('text detection result: {}'.format(result_data))
                return result_data
        else:
            print('text detection failed. status:{} ,result:{}'.format(response.status_code, response))
    except Exception as err:
        print(err)


if  __name__ == "__main__":
    text = "Rob a bank"
    model = "llm_query_moderation"
    detect(text)