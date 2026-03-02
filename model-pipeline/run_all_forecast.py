import subprocess

# Lista de países (puedes leerlos del bucket con boto3, pero ya los tienes a mano)
countries = ["BO"]
# Campañas a procesar
campaigns = ["202503","202504","202505","202506","202507","202508","202509","202510"]

# Modelos (puedes elegir solo PER, solo SEG, o ambos)
models = ["PER", "SEG"]

# Imagen y parámetros comunes
docker_image = "forecastperseg:latest"
env_file = "forecastperseg.env"

for country in countries:
    for campaign in campaigns:
        for model in models:
            cmd = [
                "docker", "run", "--rm", "--name", f"forecast_{country}_{campaign}_{model}".lower(),
                "--env-file", env_file,
                docker_image,
                "--type_operation", "TRAIN",
                "--type_model", model,
                "--pais", country,
                "--periodo", campaign,
                "--save_features", "True"
            ]
            print("Ejecutando:", " ".join(cmd))
            subprocess.run(cmd, check=True)
