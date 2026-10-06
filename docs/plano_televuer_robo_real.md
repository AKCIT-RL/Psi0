# Plano — televuer / Meta Quest 3 no teleop do robô real

Integrar o televuer ao `real/teleop` do Psi0, para teleoperar o G1 **de mão fixa** pelos
controles do Quest 3, e gravar dados aproveitáveis para treino.

Estado em 2026-10-06: levantamento feito, **nada implementado**.

> **Leia isto antes do resto.** Este plano analisa o `real/teleop` com **AMO**, que o upstream
> trata como caminho legado. O `physical-superintelligence-lab/Psi0` em `main` já suporta
> **coleta, fine-tune e deploy com SONIC** (`NVlabs/GR00T-WholeBodyControl`), com saída direta em
> LeRobot e sem conversor — ver §8. Este fork está 42 commits atrás e não tem esse caminho.
> Decidir entre os dois vem **antes** de executar qualquer coisa daqui.

---

## 1. Por que é mais fácil do que parece

O `real/teleop` do Psi0 **já é vuer**: as classes se chamam `VuerPreprocessor` e `VuerTeleop`,
e a saída do `step()` é `head_rmat, left_wrist_mat, right_wrist_mat, left_hand_mat,
right_hand_mat`. Não é enxerto de stack estranha, é troca de variante da mesma família.

E o padrão de "adicionar um headset" já existe: o `vr_pico.py` é o `vr.py` duplicado com o
receptor trocado (`PicoReceiver` no lugar do TeleVision), mantendo o `VuerPreprocessor`.

---

## 2. Decisão: usar o `TeleVuer` cru, não o `TeleVuerWrapper`

O `TeleVuerWrapper` foi escrito para adaptar o televuer ao SIMPLE, e **faz conversão de base
internamente** (`tv_wrapper.py`, "Change basis convention" nas linhas 270/287/323/334). Sai em
convenção de robô, z para cima.

O `VuerPreprocessor` do Psi0 espera **OpenXR cru** (usa `head_mat[1,3]` como altura, ou seja
y para cima) e aplica ele mesmo as conversões:

```python
head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
left_wrist_mat = fast_mat_inv(head_mat) @ left_wrist_mat @ hand2inspire_l_arm
```

Usar o wrapper converteria duas vezes. O `TeleVuer` cru entrega sem conversão — as docstrings
dizem explicitamente *"basis OpenXR Convention"*.

### Mapeamento de campos (verificado)

| Psi0 lê de `tv` | `TeleVuer` cru expõe | Shape |
|---|---|---|
| `tv.head_matrix` | `head_pose` | (4,4) OpenXR |
| `tv.left_hand` (é o punho) | `left_arm_pose` | (4,4) OpenXR |
| `tv.right_hand` | `right_arm_pose` | (4,4) OpenXR |
| `tv.left_landmarks` | `left_hand_positions` | (25,3) |
| `tv.right_landmarks` | `right_hand_positions` | (25,3) |

Só mudam os nomes. Nenhuma matriz de conversão, nenhum ajuste de sinal.

### Modo controle

Em `on_controller_move`, o televuer escreve nos **mesmos** buffers que o hand tracking usa:

```python
self.left_arm_pose_shared[:]  = event.value["left"]
self.right_arm_pose_shared[:] = event.value["right"]
```

Então `head_pose` e `left/right_arm_pose` funcionam igual nos dois modos. Os landmarks ficam
vazios — irrelevante, porque a mão é fixa. E vêm de brinde os botões: `trigger`, `triggerValue`,
`squeeze`, `squeezeValue`, `thumbstick`, `aButton`, `bButton`, por lado.

---

## 3. Mão fixa: restaurar a condicional, não arrancar código

O upstream (`xr_teleoperate/teleop/teleop_hand_and_arm.py`) torna o end-effector opcional:

```python
parser.add_argument('--ee', choices=['dex1','dex3','inspire_ftp','inspire_dfx','brainco'])
...
if args.ee == "dex3":
    from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
```

Sem `--ee`, nada de mão é importado e o braço opera normal.

O Psi0 colapsou isso: em `master_whole_body.py` o `Dex3_1_Controller` é instanciado dentro do
`if robot == "g1"`, sem condicional própria. O conserto é **restaurar a flag**, pondo atrás dela
os três pontos de uso: `hand_shm`, `hand_ctrl`, `ctrl_dual_hand`.

O gravador não quebra sem a mão: o layout é de offset fixo (`G1_sizes`: LEG 15, ARM 14,
HAND 14, depois IMU/odom), então as 14 posições de mão ficam zeradas — que é o que o SIMPLE já
faz no robô de mão fixa.

---

## 4. A gravação **está** ligada (correção)

Uma versão anterior deste plano afirmava que a gravação estava desconectada, porque o
`EpisodeWriter` não é importado em lugar nenhum. **Isso estava errado.** O `EpisodeWriter` é um
caminho alternativo abandonado; o caminho vivo é outro:

```
manager.run_command_loop()        teclas: s inicia, q encerra+merge, d descarta
   |
   +-- worker  -> AsyncWriter    -> robot_data.jsonl   (estados MEDIDOS)
   +-- master  -> IKDataWriter   -> ik_data.jsonl      (acoes COMANDADAS)
                        |
                   merger.py  casa os dois por timestamp
                        |
                    data.json   <- o que raw_to_lerobot_v2.py consome
```

O `merge_json()` indexa o `robot_data` por `time`, procura o `ik_data` correspondente por
`armtime`, e repete o último comando quando não há IK naquele instante. **A separação em dois
processos é de propósito**, e o sincronismo já está resolvido com timestamps — não é preciso
canal novo nem shared memory adicional.

### O que realmente falta verificar

1. Se o `data.json` do merge sai no envelope que o `raw_to_lerobot_v2.py` espera (chave `frames`,
   campo `image` por frame). O merge produz uma lista de entradas do `robot_data`.
2. Se o `ik_data.jsonl` carrega todos os campos do contrato de 36 dims.
3. A convenção do tronco no estado (§5) — essa continua valendo.

---

## 5. O que gravar: estado (32) e ação (36)

### O ponto de injeção certo

O `groot_inference.py` mostra onde o VLA entra no deploy: ele **não** fala com os motores,
escreve nos mesmos campos do `master` que o teleop escreveria.

```python
master.torso_roll/pitch/yaw/height = ...
master.vx / vy / vyaw / target_yaw = ...
arm_cmd = action[14:28]
```

Daí para baixo o caminho é idêntico nos dois casos: `adapter_jit.pt` -> `amo_jit.pt` ->
`pd_target` -> PD -> motores. **O VLA substitui o operador, não o controlador.**

Logo, a ação a gravar é esse comando de alto nível — **não** o `pd_target` de 29 juntas que o
AMO produz depois, que é consequência.

### Ação (36), escrita pelo master

```python
a[0:7]   = alvo mao esquerda      # zeros com mao fixa
a[7:14]  = alvo mao direita
a[14:21] = alvo braco esquerdo
a[21:28] = alvo braco direito
a[28:31] = [torso_roll, torso_pitch, torso_yaw]
a[31]    = torso_height
a[32:36] = [vx, vy, vyaw, target_yaw]
```

### Estado (32), do `robot_shm_array`

Layout (`G1_sizes`): LEG 15, ARM 14, HAND 14, depois IMU e odom.

```python
s[0:7]   = handstate[0:7]            # zeros com mao fixa
s[7:14]  = handstate[7:14]
s[14:21] = armstate[0:7]
s[21:28] = armstate[7:14]
s[28:31] = legstate[[13, 14, 12]]    # roll, pitch, yaw -- MEDIDOS
s[31]    = acao_anterior[31]         # altura: comando do frame anterior
```

**Os índices da cintura estão resolvidos** (era o item aberto): em
`robot_control/robot_arm_joints.py`, `kWaistYaw=12`, `kWaistRoll=13`, `kWaistPitch=14`, e
`LEG_STATE_SIZE=15` — a cintura são as três últimas posições do bloco de pernas.

Atenção à permutação `[13, 14, 12]`: o robô guarda yaw/roll/pitch, o contrato quer
roll/pitch/yaw. É o mesmo erro fácil que corrigimos no `vla_client.py` do IHS.

A altura é a única dimensão do estado que não é medição — comando do frame anterior, conforme o
padrão SIMPLE.

> **Descasamento conhecido no deploy:** o `groot_inference.py` lê `master.torso_*` de volta como
> estado, ou seja, comandado. Pelo padrão SIMPLE deveria ser a cintura medida, como os braços já
> são. É conserto de uma linha naquele arquivo, e está fora deste plano. Ver memória
> `eval-torso-state-semantics`.

---

## 6. Ordem de execução

1. **Restaurar a flag `--ee`** (§3). Isolado, testável sem VR e sem headset.
2. **`vr_televuer.py`** no molde do `vr_pico.py`, com o shim de nomes (§2).
3. **Mover o `img_shm` para o `manager.py`** e passar o nome aos dois processos (§4).
4. **Religar o `EpisodeWriter` no master**, no mesmo ponto do `ik_writer.write_data`, montando
   estado e ação conforme §5.
5. **Conversão ponta a ponta**: gravar um episódio curto e passar pelo `raw_to_lerobot_v2.py`.

Os passos 1 e 3 não dependem do Quest e podem ser feitos antes de qualquer coisa com o headset.

---

## 6.1 Por que o `xr_teleoperate` não serve como fonte de dados

Ele **grava** comandos de locomoção (o `replay.py` os reproduz), então a tentação de usá-lo como
atalho é real. Mas a locomoção dele é outra:

| | `xr_teleoperate` | Psi0 `real/` (deploy) |
|---|---|---|
| Controlador | `LocoClientWrapper.Move()` — **de fábrica**, alto nível | AMO (`amo_jit.pt`) |
| Comando | `vx, vy, vyaw` (thumbstick × 0,3) | `vx, vy, vyaw, target_yaw` **+ altura + rpy do tronco** |
| Tronco | não existe no comando | é parte central |

Duas incompatibilidades, e a segunda decide:

- **A resposta dinâmica difere.** O mesmo `vx` dá marcha diferente no controlador de fábrica e no
  AMO — mesma classe de problema que a `gear_wbc` vs GR00T-WBC no IHS.
- **O tronco não tem como ser gravado.** O `LocoClient.Move()` não aceita altura nem rpy; nenhum
  operador consegue comandá-los naquele modo. Quatro dims de ação ficariam constantes por
  **impossibilidade física**, e o treino ensinaria que a resposta certa ali é sempre a mesma.

É o mesmo buraco do `convert_to_psi_manipulation.py` (que chumba `[0,0,0,0.75]` e quatro zeros) —
só que a falta está na fonte, não no conversor.

**O `xr_teleoperate` continua útil como referência de padrão** (`--ee` opcional, laço único) e
para manipulação de base parada, se a tarefa for essa.

> **O mesmo argumento pesa contra o AMO.** Se o critério é alinhar coleta, treino e deploy no
> mesmo controlador, o caminho SONIC (§8) alinha com o `gear_wbc` que o SIMPLE usa no treino em
> simulação; o AMO não. A §6.1 derruba o `xr_teleoperate`, mas não eleva o AMO a melhor escolha.

---

## 7. O que não foi verificado

- **Ordem dos 25 landmarks** — irrelevante com mão fixa, mas vira problema se um dia usarem
  hand tracking. O televuer fixa 25 (padrão WebXR); o `dex_retargeting` do Psi0 usa `.shape[0]`
  genericamente, então o número passa, mas a ordenação não foi conferida.
- **Vídeo de volta para o headset: resolvido em princípio.** O Psi0 tem duas vias independentes.
  O PICO usa um caminho próprio (GStreamer x264 → TCP cru, cabeçalho de 4 bytes com o tamanho +
  NAL; `PicoIRStreamer` em `worker.py:40`), que existe porque aquele headset não entra por WebXR.
  O Vision Pro usa o Vuer, consumindo o `img_shm`. **O Quest 3 fala WebXR, então vai pela via do
  Vision Pro**: o `render_to_xr()` do `TeleVuer` cru lê o mesmo `img_shm`. Não é preciso replicar
  a complicação do GStreamer. Falta só escrever e testar.
- **`solve_whole_body_ik`** — se espera as juntas de mão no vetor de estado. Provavelmente não,
  já que resolve braço e tronco, mas não foi lido.
- **A camada física** — `safelySetMotor`, `pd_tauff`, CycloneDDS, robô pendurado. Não tem
  análogo em simulação e é onde mora o risco.

---

## 8. O caminho SONIC no upstream (não avaliado)

O `physical-superintelligence-lab/Psi0` em `main` tem:

```markdown
### Ψ₀ with SONIC
SONIC (https://github.com/NVlabs/GR00T-WholeBodyControl) is a powerful whole-body controller.
Ψ₀ now supports data collection, fine-tuning, and deployment with SONIC.
Please use our fork to avoid any compatibility issues.

git submodule update --init --recursive third_party/GR00T-WholeBodyControl
```

Coleta (`real/SONIC/README.md`):

```bash
bash real/SONIC/scripts/collect_psi0-sonic-data.sh \
  --camera-host 192.168.123.164 --network enp4s0 \
  --task-prompt '...' --dataset-name sonic_dex1 --root-output-dir "$HOME/data"
```

Controles: clique do stick direito por 0,6 s liga/desliga a política; `Left Grip + A` inicia e
encerra episódio; `Left Grip + Y` descarta.

**Saída direta em LeRobot**, sob `third_party/GR00T-WholeBodyControl/outputs/<nome>/`. Sem merge,
sem `raw_to_lerobot_v2`, sem conversor.

E o `DEPLOYMENT.md` diz que o deploy *"keeps the official four-process flow and wraps only the RTC
client so Dex1 state and action use the same mapping as collection and conversion"* — ou seja,
coleta, conversão e deploy no mesmo mapeamento.

### Por que isso provavelmente supera este plano

SONIC é a linhagem do `gear_wbc`, que é o que o SIMPLE usa. Alinharia coleta, treino em simulação
e deploy no mesmo controlador — exatamente o critério que motivou todo este documento.

### O que falta saber

- **Este fork está 42 commits atrás** do upstream. O merge tem 9 conflitos, nenhum difícil, e
  **não toca** `src/gr00t`, `scripts/prepare_simple_datasets.py` nem `baselines/gr00t-n1.7` (o
  pipeline OVX passa intacto). Os pontos de atenção são `.gitmodules` e `pyproject.toml`.
- **SONIC suporta mão fixa?** O exemplo usa `--dataset-name sonic_dex1` e o deploy fala em
  "Dex1 state and action" — Dex1 é a garra da Unitree, não mão fixa.
- Se a teleoperação SONIC usa VR, as §§1-3 (televuer, mão fixa) podem continuar valendo.
