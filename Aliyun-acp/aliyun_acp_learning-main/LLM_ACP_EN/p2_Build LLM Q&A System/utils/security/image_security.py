# coding=utf-8

from alibabacloud_green20220302.client import Client
from alibabacloud_green20220302 import models
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util import models as util_models
import json
import os
import uuid


access_key_id = os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID']
access_key_secret = os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']

config = Config(
    # The AccessKey of your Alibaba Cloud account has access to all APIs. It is recommended to use a RAM user for API access or daily operations.
    # It is strongly recommended not to save the AccessKey ID and AccessKey Secret in the project code, otherwise it may lead to AccessKey leakage, threatening the security of all resources under your account.
    # Common ways to obtain environment variables:
    # Get RAM user AccessKey ID: os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID']
    # Get RAM user AccessKey Secret: os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
    access_key_id=access_key_id,
    access_key_secret=access_key_secret,
    # Set http proxy.
    # http_proxy='http://10.10.xx.xx:xxxx',
    # Set https proxy.
    # https_proxy='https://10.10.xx.xx:xxxx',
    # The access region and address should be modified according to actual conditions.
    endpoint='green-cip.cn-beijing.aliyuncs.com'
)
client = Client(config)


def detect(image_url):

    # Create a RuntimeObject instance and set runtime parameters.
    runtime = util_models.RuntimeOptions()

    # Detection parameter construction.
    service_parameters = {
        # Publicly accessible image URL
        'imageUrl': image_url,
        # Unique data identifier
        'dataId': str(uuid.uuid1())
    }

    image_moderation_request = models.ImageModerationRequest(
        # Image detection service
        # Supported services can be referenced at: https://help.aliyun.com/document_detail/467826.html?0#p-23b-o19-gff
        service='baselineCheck_pro',
        service_parameters=json.dumps(service_parameters)
    )

    try:
        response = client.image_moderation_with_options(image_moderation_request, runtime)
        if response.status_code == 200:
            result = response.body
            # print('response success. result:{}'.format(result))
            if result.code == 200:
                result_data = result.data
                print('image detect result: {}'.format(result_data))
                return result_data
        else:
            print('image detect fail. status:{} ,result:{}'.format(response.status_code, response))
        return
    except Exception as err:
        print(err)


if __name__ == '__main__':
    image_url = "https://img.alicdn.com/imgextra/i2/O1CN01M5Cie31udzY84ppIw_!!6000000006061-2-tps-300-158.png"
    detect(image_url)