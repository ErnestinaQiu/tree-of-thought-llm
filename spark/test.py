from spark import SparkApi
#以下密钥信息从控制台获取
appid = "60bd075c"     # 填写控制台中获取的 APPID 信息
api_secret = "YmZmNWU4NTczZDg3ZGE1ZWIyMTQ3ZDdj"   # 填写控制台中获取的 APISecret 信息
api_key = "16b0409e9e80e86aad54ad1eff132db6"    # 填写控制台中获取的 APIKey 信息

# 用于配置大模型版本，默认“general/generalv2”
domain = "general"   # v1.5版本
# domain = "generalv2"    # v2.0版本
# 云端环境的服务地址
Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
# Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址


text = []

# length = 0

def getText(role, content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text

def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length

def checklen(text):
    while (getlength(text) > 8000):
        del text[0]
    return text
    

class ChatCompletion:
    def __init__(self):
        return

    @staticmethod
    def create(**kwargs):
        if "model" in kwargs.keys():
            model = kwargs["model"]
            if model not in ["spark-1.5", "spark-2.0"]:
                raise ValueError("model={} is not available".format(model))
        else:
            model = "spark-2.0"
        
        if "messages" in kwargs.keys():
            mess = kwargs["messages"]
            if len(mess) > 1:
                raise ValueError("len(mess): {} is above 1, please check again! \n mess:{}".format(len(mess), mess))
            message = mess[0]["content"]
        else:
            mess = ""
        
        if "temperature" in kwargs.keys():
            randon_rate = kwargs["temperature"]
        else:
            randon_rate = 0.5

        if model == "spark-1.5":
            domain = "general"   
            Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat" 
        elif model == "spark-2.0":
            domain = "generalv2"    
            Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat" 

        question = checklen(getText("user", message))
        SparkApi.answer = ""
        SparkApi.main(appid,api_key,api_secret,Spark_url,domain,question)
        conversation = getText("assistant", SparkApi.answer)
        res = {"content": conversation, "usage": SparkApi.record}

        return res
        
if __name__ == '__main__':
    text.clear
    while(1):
        Input = input("\n" +"usr:")
        question = checklen(getText("user",Input))
        SparkApi.answer =""
        print("spark:",end = "")
        SparkApi.main(appid,api_key,api_secret,Spark_url,domain,question)
        text = getText("assistant",SparkApi.answer) 
        print('\n')
        print(str(text))