# Mysqlc Config

# Mysql位址
host='localhost' 

# Mysql帳號
username='root' 

 # Mysql密碼
password='root'

# 資料庫名稱
database='myfirst' 

# Mysql讀取資料到資料庫的路徑 
mysqlPath='C:/Users/Jerry/Pipline_main/Garfana/cultureManualOpenGrafana-main/outcome/' 

# 站名順序
mapping_fieldname=[ 
    '1.臺灣民主紀念園區綜合氣象站',
    '2.圓山遺址綜合氣象站',
    '3.槓子寮砲台綜合氣象站',
    '4.卑南解說牌綜合氣象站',
]


# Main Config

# filename=''
# outputPath=""

# arima 或 lstm
method='arima' 

arima_hyper_parameter={
    "time":"dd/mm/YY HH:MM:SS",
}
lstm_hyper_parameter={
    "time":"dd/mm/YY HH:MM:SS",
}

# Grafana Config

# 西元年分
year="2022" 

# api 授權
apiAuthorization="eyJrIjoiU08wSDcyOFZvM1BiYVZFNHhpWUdvazFWdmVseUdMeFgiLCJuIjoia2V5IiwiaWQiOjF9" 

# Grafana 連接埠
url="http://localhost:3000" 


