import requests
import pandas as pd
import time
from datetime import datetime

# TODO esse código faz a coleta de dados, e cria o arquivo .csv
EMPRESAS_B3 = [
    "VALE3", "PETR4", "PETR3", "ITUB4", "BBDC4", "ABEV3", "BBAS3", "B3SA3", "WEGE3", "MGLU3",
    "VIVT3", "JBSS3", "GGBR4", "ELET3", "ELET6", "SUZB3", "ITSA4", "LREN3", "BRFS3", "HAPV3",
    "KLBN11", "CSAN3", "NTCO3", "RADL3", "YDUQ3", "HYPE3", "BPAC11", "TIMS3", "EQTL3", "CIEL3",
    "ENGI11", "BRML3", "MULT3", "UGPA3", "CVCB3", "CCRO3", "MRVE3", "BEEF3", "MRFG3", "ENBR3",
    "SULA11", "ALPA4", "BRAP4", "EMBR3", "GOAU4", "PSSA3", "SANB11", "SBSP3", "TAEE11", "TRPL4",
    "VVAR3", "TOTS3", "COGN3", "AZUL4", "LAME4", "BTOW3", "CMIG4", "AMER3", "ASAI3", "BBDC3",
    "BBSE3", "BPAN4", "BRKM5", "BRPR3", "CASH3", "CPFE3", "CRFB3", "CSNA3", "CYRE3", "DXCO3",
    "ECOR3", "EGIE3", "ENEV3", "EZTC3", "FLRY3", "GOLL4", "GRND3", "IRBR3", "JHSF3", "LIGT3",
    "LOGG3", "LWSA3", "MILS3", "MOVI3", "NEOE3", "PETZ3", "POSI3", "PRIO3", "QUAL3", "RAIL3",
    "RDOR3", "RENT3", "RRRP3"
]


class Dados:
    @staticmethod
    def get_stock_info(stock_symbol):
        stock_symbol += ".SA"  # Adiciona o sufixo usado na API
        url = f"https://brapi.dev/api/quote/{stock_symbol}?range=3mo&interval=1d&token=xBm92JdYfrv7MhRGKRM4Sy"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Levanta exceção para erros HTTP
            data = response.json()

            if 'results' in data and data['results']:
                stock_data = data['results'][0]
                historico = []

                for day in stock_data.get('historicalDataPrice', []):
                    timestamp = day.get('date')
                    if timestamp is None:
                        continue

                    date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
                    open_price = day.get('open', None)
                    close_price = day.get('close', None)
                    high_price = day.get('high', None)
                    low_price = day.get('low', None)
                    volume = day.get('volume', None)

                    # Calcula variação apenas se houver valores válidos
                    if open_price and close_price:
                        variation = ((close_price - open_price) / open_price) * 100
                    else:
                        variation = None

                    historico.append([stock_symbol, date, open_price, close_price, high_price, low_price, volume, variation])

                return historico
            else:
                print(f"[Aviso] Nenhum dado encontrado para {stock_symbol}.")
                return []
        except requests.RequestException as e:
            print(f"[Erro] Falha ao buscar dados de {stock_symbol}: {e}")
            return []


class Coleta:
    @staticmethod
    def salvar_csv():
        all_data = []

        for empresa in EMPRESAS_B3:
            print(f"Coletando dados de {empresa}...")
            dados = Dados.get_stock_info(empresa)
            all_data.extend(dados)

            time.sleep(1)  # Evita exceder limite da API

        if all_data:
            df = pd.DataFrame(all_data, columns=["Empresa", "Data", "Abertura", "Fechamento", "Máxima", "Mínima", "Volume", "Variação (%)"])
            df.to_csv("dados_b3.csv", index=False, encoding="utf-8")
            print("Arquivo 'dados_b3.csv' salvo com sucesso!")
        else:
            print("Nenhum dado foi coletado.")


if __name__ == "__main__":
    Coleta.salvar_csv()
