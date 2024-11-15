
apt-get update
apt-get upgrade -y

apt-get install -y docker.io docker-compose nginx certbot python3-certbot-nginx

cp nginx.conf /etc/nginx/nginx.conf
systemctl restart nginx

# Configurar SSL con Certbot (reemplaza con tu dominio)
certbot --nginx -d tu-dominio.digitalocean.com

docker-compose -f docker-compose.yml up -d --build

docker ps