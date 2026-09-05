wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = [[
    {
       "model": "./model/qwen2_5-1_5b-instruct",
       "messages": [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "Please tell me how many gold medals the Chinese team won in total at the 2008 Beijing Olympics?"}
       ]
   }
]]