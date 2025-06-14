```python
import http.client
import json


class APIData:

    @staticmethod
    def generate_data() -> None:
        conn = http.client.HTTPSConnection(
            "ojyolu1o2k.execute-api.us-east-1.amazonaws.com"
        )
        payload = json.dumps({
            "layer": "app",
            "system": "FORECASTDSSPERSEG",
            "uuid": "",
            "country": "PE",
            "idProcess": "DIG_PERSEG_200",
            "sourceevent": "API-SNOW",
            "message": {
                "P_ANNIO_CAMPANA": "202501"
            },
            "domain": [
                "AAN_ForecastDssPerSeg_GenTrainDataSetInOneShotForPer",
                "AAN_ForecastDssPerSeg_GenTrainDataSetInOneShotForSeg"
            ],
            "domainType": "DELIVERY"
        })
        headers = {'Content-Type': 'application/json'}
        conn.request(
            "GET",
            "/default/AWLFUSE1EDLBDEV009_PipelineEventDriver",
            payload,
            headers
        )
        res = conn.getresponse()
        data = res.read()
        print(data.decode("utf-8"))
```
