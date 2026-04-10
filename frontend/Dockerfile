FROM node:22-alpine AS base

WORKDIR /app

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./

# ─────────────────────────────────────────────────────────────────────────────
FROM base AS dev
EXPOSE 5173
# --host 0.0.0.0 makes Vite reachable from outside the container (docker-compose port mapping)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# ─────────────────────────────────────────────────────────────────────────────
FROM base AS builder

ENV NODE_ENV=production
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build
# ─────────────────────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS prod

# Drop default config so only our app.conf is active
RUN rm /etc/nginx/conf.d/default.conf
COPY docker/nginx/nginx.conf /etc/nginx/conf.d/app.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
