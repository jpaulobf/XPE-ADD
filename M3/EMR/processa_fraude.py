from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, lit, lag, split
from pyspark.sql.window import Window
from math import radians, sin, cos, sqrt, atan2

# Função para calcular a distância em km entre duas coordenadas (lat, lon) usando a fórmula de Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Raio da Terra em quilômetros
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Inicializar a sessão do Spark
spark = SparkSession.builder.appName("Fraud Detection").getOrCreate()

# Caminho do arquivo no bucket S3
input_path = "s3://your-bucket-name/compras.log"  # Substitua pelo caminho do seu bucket S3

# Carregar o arquivo de logs
df = spark.read.csv(input_path, header=True, inferSchema=True)

# Exibir o esquema dos dados
df.printSchema()

# Supondo que a última coluna seja "Geo_Localizacao", que possui latitudes e longitudes separadas por vírgula (ex: "-23.5505,-46.6333")
# Vamos dividir esta coluna em duas novas colunas: "Latitude" e "Longitude"
df = df.withColumn("Latitude", split(col("Geo_Localizacao"), ",").getItem(0).cast("double"))
df = df.withColumn("Longitude", split(col("Geo_Localizacao"), ",").getItem(1).cast("double"))

# Janela para verificar compras consecutivas por CPF
window_spec = Window.partitionBy("CPF").orderBy("Data_Hora")

# Converter a coluna de Data_Hora para o formato timestamp
df = df.withColumn("Data_Hora", unix_timestamp(col("Data_Hora"), "yyyy-MM-dd HH:mm:ss").cast("timestamp"))

# Calcular o tempo entre compras consecutivas
df = df.withColumn("Tempo_Entre_Compras", unix_timestamp(col("Data_Hora")) - lag(unix_timestamp(col("Data_Hora"))).over(window_spec))

# Adicionar flag para compras impossíveis com base na distância e tempo de compra
df = df.withColumn(
    "Risco_Fraude",
    (col("Compra_Virtual") == lit(False)) & (col("Tempo_Entre_Compras") < 3600)  # Compras físicas em cidades distantes no mesmo intervalo
)

# Função para calcular a distância entre duas compras consecutivas
def calc_distance(lat1, lon1, lat2, lon2):
    if lat1 is not None and lat2 is not None:
        return haversine(lat1, lon1, lat2, lon2)
    return None

# Calcular a distância entre as compras consecutivas (em km)
df = df.withColumn(
    "Distancia_Cidades",
    calc_distance(col("Latitude"), col("Longitude"), lag(col("Latitude")).over(window_spec), lag(col("Longitude")).over(window_spec))
)

# Condição de risco (compras físicas em cidades > 100 km no mesmo intervalo de tempo)
df = df.withColumn(
    "Suspeita_Fraude",
    (col("Distancia_Cidades") > 100) & (col("Tempo_Entre_Compras") < 3600)  # Distância maior que 100km e intervalo menor que 60min
)

# Filtrar compras suspeitas
fraudes = df.filter(col("Suspeita_Fraude") == lit(True))

# Salvar resultado em um bucket S3
output_path = "s3://your-bucket-name/suspect_purchases/"  # Substitua pelo seu bucket S3
fraudes.write.csv(output_path, header=True, mode="overwrite")

# Exibir as fraudes encontradas
fraudes.show(truncate=False)

# Encerrar sessão do Spark
spark.stop()
