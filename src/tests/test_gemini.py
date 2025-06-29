import google.generativeai as genai

if __name__ == '__main__':
    # 注意gemini的sdk必须制定rest协议调用，否则会默认走grpc造成报错
    genai.configure(
        api_key='sk-HoCIo0kiA6xoIWWfybIvu2T17TCtVhGM1HGh6kk9tQy5wMi7',
        transport="rest",
        client_options={"api_endpoint": "https://api.openai-proxy.org/google"},
    )
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Say Hello")
    print(response.text)