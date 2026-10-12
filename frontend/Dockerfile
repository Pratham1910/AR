# Pinned by digest (Node 20.20.2) so a rebuild starts from the same image and,
# once it has been pulled, does not need to contact Docker Hub: with a bare
# tag BuildKit asks the registry on every build and fails when it is offline.
# To update: `docker pull node:20-slim`, then
# `docker image inspect node:20-slim --format '{{index .RepoDigests 0}}'`.
FROM node:20-slim@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]
