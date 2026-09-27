# Despliegue en Railway

Guia paso a paso para publicar el servicio HTTP del normalizador
(`cali_address.api.main:app`) en Railway.

Todas las cifras de memoria, tamano de imagen y tiempo de arranque de este
documento estan **medidas** sobre la imagen real (`docker build` +
`docker run` + `docker stats`), no estimadas. La seccion
[Memoria y plan recomendado](#c-memoria-medida-y-plan-recomendado) dice como se
midieron.

---

## a. Prerrequisitos

1. Una cuenta de Railway (<https://railway.app>).
2. La CLI de Railway, para el camino recomendado:

   ```bash
   npm i -g @railway/cli
   railway --version
   railway login
   ```

3. Docker Desktop, solo si quiere probar la imagen localmente antes de subirla.
   No es necesario para desplegar: Railway construye la imagen del lado del
   servidor.
4. Los seis artefactos de servicio presentes en `artifacts/`:
   `model.pt`, `catastro_emb.pt`, `catastro_docs.parquet`, `tuning.json`,
   `gazetteer.pkl` y `reliability.json` (modelo de confiabilidad calibrada, ~3 KB,
   generado con `scripts/fit_reliability.py`; sin el, `confiabilidad` sale nula). Se generan con el pipeline de entrenamiento y calibracion del
   proyecto; el despliegue no los reentrena ni los descarga. Ver
   [docs/model-artifacts.md](docs/model-artifacts.md) para saber cuales estan en git y
   como regenerar `catastro_emb.pt` (161 MB, no versionado).

### Paso obligatorio antes de cualquier despliegue

```bash
python deploy/prepare_artifacts.py
```

Copia **solo** esos seis archivos de `artifacts/` (415 MB) a
`deploy/artifacts/` (199 MB), que es el directorio que la imagen copia. Es
idempotente e imprime el total:

```
preparing serving artifacts: .../artifacts -> .../deploy/artifacts
  copying     model.pt                    12.9 MB
  copying     catastro_emb.pt            161.3 MB
  copying     catastro_docs.parquet       20.3 MB
  copying     tuning.json                  5.3 KB
  copying     gazetteer.pkl                4.5 MB
  copying     reliability.json             3.0 KB
total: 199.0 MB across 6 files
```

---

## b. Dos caminos de despliegue

### Camino 1 (recomendado): `railway up`, sin git

Sube el directorio local directamente. **No requiere repositorio git**, que es
precisamente lo que hace viable este servicio: `catastro_emb.pt` pesa 161 MB, por
encima del limite duro de 100 MB por archivo de GitHub, asi que los artefactos no
pueden viajar por git sin Git LFS (ver Camino 2).

```bash
# 1. preparar los artefactos de servicio (199 MB)
python deploy/prepare_artifacts.py

# 2. autenticarse y crear/vincular el proyecto
railway login
railway init            # o: railway link   si el proyecto ya existe

# 3. variables de entorno del servicio
railway variables --set NORMALIZER_DEVICE=cpu \
                  --set ARTIFACTS_DIR=/app/artifacts \
                  --set BASEMAPS_DIR=/app/basemaps \
                  --set DOCS_DIR=/app/docs \
                  --set MAX_UPLOAD_MB=25 \
                  --set MAX_ROWS=20000 \
                  --set WARMUP_TIMEOUT_S=300

# 4. desplegar
railway up --no-gitignore

# 5. exponer el servicio en una URL publica
railway domain
```

**Por que `--no-gitignore`.** `deploy/artifacts/` esta en `.gitignore` a
proposito (199 MB de binarios que nunca deben entrar en la historia de git).
`railway up` respeta `.gitignore` por omision, lo que dejaria los artefactos
fuera del paquete y la imagen fallaria al construir. Con `--no-gitignore`, el
archivo **`.railwayignore`** pasa a ser la lista de exclusiones vigente, y ahi
estan listados de forma explicita `context/`, `outputs/`, `artifacts/`,
`scripts/`, `tests/` y el notebook.

> **Verificacion obligatoria antes de confiar en el primer despliegue.**
> Mire el tamano que reporta la subida en el log de `railway up`. Debe rondar los
> **205 MB** (199 MB de artefactos + 12 MB de basemaps + el codigo). Si ve
> **250 MB o mas**, `context/` viajo con el paquete: **cancele el despliegue**,
> revise `.railwayignore` y vuelva a intentar. `context/` son 47 MB de datos
> personales de ciudadanos y no debe salir de la maquina local.
>
> Si prefiere no depender de la semantica de los archivos de exclusion, la
> alternativa de riesgo cero es subir desde una copia limpia:
>
> ```bash
> # desde el directorio padre del proyecto
> cp -r normalizador_direcciones /tmp/deploy_limpio
> rm -rf /tmp/deploy_limpio/context /tmp/deploy_limpio/outputs \
>        /tmp/deploy_limpio/artifacts /tmp/deploy_limpio/direcciones_ANN.ipynb
> cd /tmp/deploy_limpio && railway up --no-gitignore
> ```
>
> (`deploy/artifacts/` se conserva en la copia; `artifacts/` se borra.)

Para desplegar de nuevo tras un cambio de codigo:

```bash
python deploy/prepare_artifacts.py     # no copia nada si ya estan al dia
railway up --no-gitignore
```

### Camino 2: conectado a GitHub

Railway construye en cada `push`. **Este camino lo ejecuta usted**: este
repositorio todavia no es un repositorio git y los comandos de git no se han
ejecutado por usted.

```bash
git init
git add .
git commit -m "feat: servicio HTTP de normalizacion de direcciones"
git branch -M main
git remote add origin git@github.com:<usuario>/<repo>.git
git push -u origin main
```

Luego, en el panel de Railway: **New Project -> Deploy from GitHub repo**,
elija el repositorio y configure las mismas variables de entorno del Camino 1.
`railway.toml` ya indica el builder `DOCKERFILE` y el health check.

**Limitacion real de este camino.** `.gitignore` excluye `artifacts/` y
`deploy/artifacts/`, asi que el repositorio **no lleva los artefactos** y
`docker build` fallara con `deploy/artifacts/: not found`. `git add` de esos
archivos tampoco es una salida: GitHub rechaza `catastro_emb.pt` por superar los
100 MB por archivo. Para usar GitHub hay que resolver antes como llegan los
199 MB al build, con una de estas tres opciones:

1. **Git LFS.** `git lfs install && git lfs track "deploy/artifacts/*"`, quitar
   `deploy/artifacts/` de `.gitignore` y hacer commit. Verifique que su plan de
   GitHub tenga cuota de LFS y que Railway haga `git lfs pull` en el build; si no
   lo hace, el build recibira punteros LFS en lugar de los binarios.
2. **Descarga en tiempo de build.** Publicar los seis archivos en
   almacenamiento de objetos (S3, R2, un release de GitHub) y agregar al
   `Dockerfile`, antes del `COPY deploy/artifacts/`, un `RUN` que los descargue
   con `curl` usando una variable `ARTIFACTS_BASE_URL`. **No esta implementado en
   este repositorio** porque no hay un almacenamiento definido; si elige esta
   via, hay que agregar ese paso y probarlo.
3. **Usar el Camino 1**, que es la recomendacion, y reservar GitHub solo para el
   codigo.

No sirve poner `prepare_artifacts.py` como *build command* de Railway: el script
lee de `artifacts/`, que nunca llega al servidor de build.

---

## c. Memoria medida y plan recomendado

Medido con la imagen real en esta maquina (`python:3.12-slim`, torch CPU,
`docker stats` sobre el contenedor, que reporta el total del cgroup):

| Momento | Memoria residente |
| --- | --- |
| Recien listo (`/health` responde 200) | **765 MiB** |
| En reposo, tras algunas peticiones | **821 MiB** |
| Despues de un lote de 2 000 direcciones | **1 014 MiB** |
| Despues de un lote de 10 000 direcciones | **~1,0 GiB** (1 024 MiB) |

Otras cifras medidas:

- **Imagen: 2,33 GB en disco / 609 MB comprimida.** Son dos cifras distintas y
  hacen falta las dos: 609 MB es lo que se transfiere al registro y lo que la
  plataforma descarga; 2,33 GB es lo que ocupa descomprimida en el disco del
  host. Desglose por capa (`docker history`): torch CPU **930 MB**, el resto de
  `requirements.txt` **426 MB**, artefactos de servicio **209 MB**, imagen base
  mas `apt` **~147 MB**, basemaps **11,7 MB**, codigo **0,7 MB**. Dentro del
  contenedor, `du -sh /` da **1,7 GB**, de los cuales 1,3 GB son
  `site-packages`.
- **Arranque en frio hasta `/health` 200:** **3,97 s**, incluyendo el
  `docker run`, con la imagen ya presente en el host. El modelo se carga en un
  hilo aparte, asi que el puerto queda disponible de inmediato y `/health`
  responde 503 con `{"status":"loading"}` mientras carga. Ese numero **no**
  incluye la descarga de la imagen: en el primer despliegue, o tras un reinicio
  en un host sin cache, hay que sumar la transferencia de los 609 MB
  comprimidos.
- **Construccion:** ~2 min en frio (la descarga de la rueda de torch domina) y
  **4,1 s** en una reconstruccion que solo cambia `src/`, porque las capas de
  `pip install` quedan cacheadas por delante de los `COPY`.
- **Rendimiento:** 2 000 direcciones en ~4 s y 10 000 en ~29 s sobre CPU, con un
  solo worker.

> **Como medir el tamano de imagen, porque `docker inspect` engana.** Con el
> almacen de imagenes de containerd (`io.containerd.snapshotter.v1`, el de Docker
> Desktop actual), `docker image inspect --format '{{.Size}}'` devuelve **609 MB**:
> es el tamano del contenido comprimido, no la huella en disco. Use estos tres:
>
> ```bash
> docker images cali-normalizador          # columna DISK USAGE -> 2.33GB
> docker system df -v | grep cali          # columna UNIQUE SIZE -> 2.334GB
> docker history cali-normalizador         # desglose por capa
> ```

De donde sale ese ~1 GB: 330 387 embeddings en fp16 (169 MB), el checkpoint del
modelo (13 MB), `catastro_docs.parquet` expandido a un DataFrame de pandas con
las columnas de direccion, predial, manzana y coordenadas (varios cientos de MB
como objetos de Python), el gazetteer con los poligonos de shapely, y el runtime
de torch CPU.

### Recomendacion

**Provisione el servicio con al menos 2 GB de RAM.** El pico medido es de ~1 GiB
y el margen restante absorbe el DataFrame temporal de un archivo grande.

**Y al menos 4 GB de disco.** La imagen ocupa 2,33 GB descomprimida; sumando el
contexto de build y una version anterior conservada para un posible rollback, un
limite de disco de 2 GB o menos falla al desplegar. Este consumo es independiente
de la RAM: confirme ambos limites en el panel de Railway.

**Consecuencias del tamano de la imagen en los despliegues**, ninguna de las
cuales afecta la cifra de RAM de arriba:

- El primer despliegue y cualquier arranque en un host sin cache tienen que
  descargar 609 MB comprimidos antes de que el contenedor exista. Los 3,97 s
  medidos son solo el arranque del proceso; el tiempo real hasta el primer
  `/health` 200 en un despliegue nuevo sera de minutos, no de segundos. Suba
  `healthcheckTimeout` en `railway.toml` (ya esta en 300 s) antes que reducirlo.
- Un servicio que escala a cero o se reinicia con frecuencia paga esa descarga
  cada vez. Si le importa la latencia del primer pedido, mantenga el servicio
  siempre encendido en lugar de confiar en un arranque por demanda.
- El peso viene casi todo de torch CPU (930 MB) y del resto de las dependencias
  cientificas (426 MB), no de los artefactos (209 MB). Recortar artefactos no
  reducira la imagen de forma apreciable; lo unico con impacto real seria
  sustituir torch por una ejecucion del codificador en ONNX Runtime, que es un
  cambio de alcance mucho mayor y no forma parte de este despliegue.

**Advertencia concreta: cualquier limite de memoria de 1 GB o menos no alcanza.**
El servicio ya ocupa 765 MiB apenas termina de cargar, y el primer lote grande lo
lleva por encima de 1 GiB: seria un OOM kill en el arranque o en la primera
peticion real. Antes de desplegar, confirme en el panel de Railway el limite de
memoria vigente para su plan y su servicio (Railway ha cambiado esos limites
varias veces, por lo que este documento no afirma cifras por plan) y, si es de
1 GB o menos, suba de plan antes de intentar el despliegue.

No aumente `numReplicas` ni el numero de workers de uvicorn para ganar
concurrencia sin subir la memoria: **cada worker carga su propia copia del modelo
y de los 330 000 embeddings**, asi que dos workers duplican el consumo. Para mas
throughput, escale horizontalmente con mas memoria por instancia, no con mas
workers dentro de la misma.

---

## d. Como la imagen recibe solo los 6 artefactos de servicio

`artifacts/` pesa 415 MB: tensores de entrenamiento (`train_data.npz`,
`spatial_negatives.npy`), parquets de evaluacion, la cache del geocodificador
IDESC, barridos de umbrales y logs. Nada de eso hace falta para servir.

El mecanismo son tres piezas que se refuerzan entre si:

1. **`deploy/prepare_artifacts.py`** copia exactamente `model.pt`,
   `catastro_emb.pt`, `catastro_docs.parquet`, `tuning.json`, `gazetteer.pkl` y `reliability.json` a
   `deploy/artifacts/` (199 MB) e imprime el total.
2. **`.dockerignore`** excluye `artifacts/` por completo, de modo que el
   directorio grande ni siquiera entra en el contexto de build.
3. **`Dockerfile`** hace `COPY deploy/artifacts/ ./artifacts/` y despues verifica,
   en tiempo de build, que los seis archivos y los basemaps existan. Un
   despliegue al que le falte un artefacto falla al construir, no en produccion.

Verificado sobre la imagen construida:

```console
$ docker run --rm --entrypoint sh cali-normalizador -c "ls /app && ls -la /app/artifacts"
artifacts
basemaps
docs
requirements.txt
src

total 203832
-rwxr-xr-x 1 root root  21279102 catastro_docs.parquet
-rwxr-xr-x 1 root root 169159756 catastro_emb.pt
-rwxr-xr-x 1 root root   4732141 gazetteer.pkl
-rwxr-xr-x 1 root root  13527959 model.pt
-rwxr-xr-x 1 root root      5424 tuning.json

$ docker run --rm --entrypoint sh cali-normalizador -c "ls /app/context /app/outputs /app/scripts"
ls: cannot access '/app/context': No such file or directory
ls: cannot access '/app/outputs': No such file or directory
ls: cannot access '/app/scripts': No such file or directory
```

`torch` no esta en `requirements.txt`: el `Dockerfile` lo instala aparte con
`pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu`,
porque la rueda CPU vive en un indice propio y `requirements.txt` no puede
expresar ese indice sin redirigir la resolucion de todo lo demas. La rueda CPU
pesa **930 MB** instalada (medido con `docker history`) frente a los **3,08 GB**
de la version CUDA medidos en el entorno de desarrollo de este proyecto: sigue
siendo el componente mas grande de la imagen, pero evita 2,1 GB. El codigo es agnostico
del dispositivo (`NORMALIZER_DEVICE`, por omision `cpu`).

### Probar la imagen localmente (opcional pero recomendado)

```bash
python deploy/prepare_artifacts.py
docker build -t cali-normalizador .
docker run -d --name cali-test -p 8000:8000 -e PORT=8000 cali-normalizador

curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/api/v1/normalize-address \
  -H "Content-Type: application/json" \
  -d '{"addresses": ["Carrera1#9-80", "no es una direccion"]}'

docker stats cali-test --no-stream        # comprobar la memoria en su maquina
docker rm -f cali-test
```

---

## e. Verificar el despliegue

```bash
# 1. el servicio esta vivo y el modelo cargado
curl -s https://<app>.up.railway.app/health
# {"status":"ok","model_loaded":true,"device":"cpu","catastro_size":330387,
#  "gazetteer_loaded":true,"threshold":0.73,"detail":null}

# 2. una direccion que debe coincidir y otra que no
curl -s -X POST https://<app>.up.railway.app/api/v1/normalize-address \
  -H "Content-Type: application/json" \
  -d '{"addresses": ["Carrera1#9-80", "no es una direccion"]}'

# 3. la interfaz de carga y la documentacion
curl -s -o /dev/null -w "%{http_code}\n" https://<app>.up.railway.app/
curl -s -o /dev/null -w "%{http_code}\n" https://<app>.up.railway.app/docs
curl -s -o /dev/null -w "%{http_code}\n" https://<app>.up.railway.app/docs-integracion

# 4. la forma Esri
curl -s -G "https://<app>.up.railway.app/arcgis/rest/services/CaliNormalizador/GeocodeServer/findAddressCandidates" \
  --data-urlencode "SingleLine=Carrera1#9-80" --data-urlencode "f=json"
```

Qué esperar en cada caso:

- `/health` con **503** y `{"status":"loading"}` justo despues de desplegar es
  normal: el modelo todavia se esta cargando. Vuelva a consultar.
- `/health` con **503** y `{"status":"error", "detail": "..."}` significa que la
  carga fallo. El `detail` dice cual artefacto falta o no se pudo leer.
- Si el contenedor se reinicia una y otra vez sin llegar a responder, revise la
  memoria del plan antes que cualquier otra cosa (seccion c).

Logs y estado:

```bash
railway logs
railway status
railway variables
```

---

## f. Privacidad: nunca despliegue con `context/` ni `outputs/`

`context/` contiene **datos reales de ciudadanos**:

- el registro de inmuebles asegurados (Fasecolda);
- el Registro Unico de Damnificados del Valle del Cauca, **con nombres y numeros
  de documento**;
- reportes de inspeccion de campo con direcciones y evaluaciones estructurales.

`outputs/` contiene resultados derivados de esos archivos y por lo tanto tiene la
misma sensibilidad.

Reglas, sin excepciones:

1. **Nunca los suba a git.** Estan en `.gitignore`. Un commit que los agregue
   filtra datos personales de forma irreversible: la historia de git los conserva
   incluso despues de borrarlos en un commit posterior.
2. **Nunca los incluya en una imagen.** Estan en `.dockerignore`, y la seccion d
   muestra la verificacion de que no estan en `/app`.
3. **Nunca los suba a Railway.** Estan en `.railwayignore`. Como
   `railway up --no-gitignore` desactiva `.gitignore`, ese archivo es la unica
   barrera: verifique el tamano de la subida como indica el Camino 1.
4. **Ninguna ruta por omision del servicio los referencia.** El servicio solo lee
   `ARTIFACTS_DIR` y `BASEMAPS_DIR`. Los archivos que llegan por
   `multipart/form-data` se procesan en memoria y se descartan al terminar la
   peticion: nada se escribe en disco.
5. Si necesita normalizar esos archivos, hagalo **localmente** con
   `scripts/normalizar.py`, o subalos a traves de la interfaz web del servicio ya
   desplegado, que los procesa en memoria sin persistirlos. No los copie al
   paquete de despliegue.

Antes de un `railway up`, una comprobacion rapida:

```bash
# debe imprimir los dos directorios como ignorados
git check-ignore -v context outputs artifacts deploy/artifacts 2>/dev/null || \
  echo "aun no hay repositorio git; verifique .railwayignore a mano"
```
