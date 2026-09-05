# coding=utf-8
# python version >= 3.6
import uuid
import oss2
import time


# Whether the service is deployed on VPC
is_vpc = False
# File upload token endpoint->token
token_dict = dict()
# Upload file client
bucket = None
# Access region and address should be modified according to actual conditions.
# end_point = 'green-cip.cn-beijing.aliyuncs.com'


# Create a file upload client
def create_oss_bucket(is_vpc, upload_token):
    global token_dict
    global bucket
    auth = oss2.StsAuth(upload_token.access_key_id, upload_token.access_key_secret, upload_token.security_token)

    if (is_vpc):
        end_point = upload_token.oss_internal_end_point
    else:
        end_point = upload_token.oss_internet_end_point
    # Note: The instantiated bucket should be reused as much as possible to avoid repeated connections and improve detection performance.
    bucket = oss2.Bucket(auth, end_point, upload_token.bucket_name)


# Upload file
def upload_file(file_name, upload_token):
    create_oss_bucket(is_vpc, upload_token)
    object_name = upload_token.file_name_prefix + str(uuid.uuid1()) + '.' + file_name.split('.')[-1]
    bucket.put_object_from_file(object_name, file_name)
    return object_name


# Get the token for file upload
def get_token(client, endpoint):
    upload_token = token_dict.setdefault(endpoint, None)
    if (upload_token == None) or int(upload_token.expiration) <= int(time.time()):
        response = client.describe_upload_token()
        upload_token = response.body.data
        token_dict[endpoint] = upload_token
    return upload_token

def get_region_id_from_endpoint(endpoint):
    # Remove the protocol part of the endpoint (if it exists), and process it uniformly in public network form to extract the region.
    endpoint = endpoint.split('//')[-1]  # Remove possible protocol prefix
    if endpoint.endswith('aliyuncs.com'):  # Public network Endpoint format processing
        region_id = endpoint.split('.')[0].split('-')[-1]
    elif endpoint.endswith('internal.aliyuncs.com'):  # VPC internal network Endpoint format processing
        region_id = endpoint.split('.')[0].split('-')[-2]
    else:
        raise ValueError("Unsupported endpoint format.")
    return region_id


if __name__ == "__main__":
    end_point = 'green-cip.cn-beijing.aliyuncs.com'
    # Call this function in your function to get RegionId
    oss_region_id = get_region_id_from_endpoint(end_point)
    print(f"The OSS RegionId is: {oss_region_id}")