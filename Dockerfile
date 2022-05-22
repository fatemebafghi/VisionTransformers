    #coercing to Unicode: need string or buffer, NoneType found
    FROM python3

    RUN mkdir -p /usr/src/VisionTransformer
    WORKDIR /usr/src/VisionTransformer
    COPY ./requirements.txt ./requirements.txt
    RUN pip install -r requirements.txt
    COPY ./VIT /usr/src/VIT
    WORKDIR /usr/src/VIT
    COPY ./config.json ./config.json
    RUN mkdir -p logs
    CMD ["/usr/bin/python3","main.py","-c","config.json"]
