#!/usr/bin/env bash
# Copyright 2026 SmallAIOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates the certificate-chain cross-vector corpus
# (tls-tcp-client-v1 task 5.8). The committed *.der files are the
# corpus; this script exists so the corpus is reproducible and
# auditable, not because regeneration is routine.
#
# Requires OpenSSL >= 3.2 (RSA-PSS -sigopt on the x509 app). On macOS
# the system LibreSSL will NOT work — point OPENSSL at Homebrew's:
#
#   OPENSSL=/opt/homebrew/opt/openssl@3/bin/openssl ./gen_corpus.sh
#
# Verifier contract the corpus targets (tls-client/src/cert/):
#   - leaf: SAN mandatory, extendedKeyUsage=serverAuth mandatory
#   - CA links: basicConstraints CA:TRUE (+ keyCertSign if keyUsage present)
#   - signatures: Ed25519, ecdsa-with-SHA256, or RSASSA-PSS with
#     SHA-256 / MGF1-SHA-256 / saltlen 32 (sLen == hLen). PKCS#1 v1.5
#     parses but is refused at verify time by policy.
#   - replay tests inject now_unix = 1_790_000_000 (2026-09-22), so
#     "-days 3650" chains are in-window and "-days 1" chains are the
#     expired case.

set -euo pipefail
OPENSSL=${OPENSSL:-openssl}
cd "$(dirname "$0")"

ver=$($OPENSSL version)
case "$ver" in
  OpenSSL\ 3.*) ;;
  *) echo "need OpenSSL 3.x (got: $ver) — set OPENSSL=..." >&2; exit 1 ;;
esac

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
serial=1000

# ── key + signing helpers ───────────────────────────────────────────
genkey() { # genkey <path> <ec|ec384|rsa2048|rsa3072|rsa4096|ed25519>
  case "$2" in
    ec)      $OPENSSL genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out "$1" 2>/dev/null ;;
    ec384)   $OPENSSL genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-384 -out "$1" 2>/dev/null ;;
    rsa2048) $OPENSSL genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$1" 2>/dev/null ;;
    rsa3072) $OPENSSL genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out "$1" 2>/dev/null ;;
    rsa4096) $OPENSSL genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:4096 -out "$1" 2>/dev/null ;;
    ed25519) $OPENSSL genpkey -algorithm ED25519 -out "$1" 2>/dev/null ;;
  esac
}

# Signature options for the *signing* key's algorithm. RSA always
# signs PSS/SHA-256/salt32 except the explicit v1.5 bad case.
sigopts() { # sigopts <ec|ec384|rsa*|ed25519|v15>
  case "$1" in
    ec)      echo "-sha256" ;;
    ec384)   echo "-sha384" ;;
    rsa*)    echo "-sha256 -sigopt rsa_padding_mode:pss -sigopt rsa_pss_saltlen:32 -sigopt rsa_mgf1_md:sha256" ;;
    ed25519) echo "" ;;
    v15)     echo "-sha256" ;;
  esac
}

mkroot() { # mkroot <name> <keytype> -> $work/<name>.key/.pem + ./<name>.der
  local name=$1 kt=$2
  genkey "$work/$name.key" "$kt"
  # shellcheck disable=SC2046
  $OPENSSL req -x509 -new -key "$work/$name.key" -subj "/CN=$name" \
    -days 3650 $(sigopts "$kt") \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out "$work/$name.pem" 2>/dev/null
  $OPENSSL x509 -in "$work/$name.pem" -outform DER -out "$name.der"
}

# mkcert <name> <keytype> <issuer> <issuer-keytype> <kind> [san] [sigkt]
#   kind:  leaf | ca | leaf_v15 (leaf PKCS#1 v1.5-signed) | ...
#   san:   for leaves, the subjectAltName value (e.g. DNS:a.corpus.test)
#   sigkt: optional sigopts() override (e.g. ec384 to force SHA-384)
mkcert() {
  local name=$1 kt=$2 issuer=$3 ikt=$4 kind=$5 san=${6:-} sigkt=${7:-}
  genkey "$work/$name.key" "$kt"
  $OPENSSL req -new -key "$work/$name.key" -subj "/CN=$name" -out "$work/$name.csr" 2>/dev/null
  local ext="$work/$name.ext" so days=3650
  case "$kind" in
    leaf|leaf_v15)
      printf 'basicConstraints=CA:FALSE\nkeyUsage=digitalSignature\nextendedKeyUsage=serverAuth\nsubjectAltName=%s\n' "$san" > "$ext" ;;
    leaf_short)
      days=1
      printf 'basicConstraints=CA:FALSE\nkeyUsage=digitalSignature\nextendedKeyUsage=serverAuth\nsubjectAltName=%s\n' "$san" > "$ext" ;;
    leaf_nosan)
      printf 'basicConstraints=CA:FALSE\nkeyUsage=digitalSignature\nextendedKeyUsage=serverAuth\n' > "$ext" ;;
    leaf_noeku)
      printf 'basicConstraints=CA:FALSE\nkeyUsage=digitalSignature\nsubjectAltName=%s\n' "$san" > "$ext" ;;
    ca)
      printf 'basicConstraints=critical,CA:TRUE\nkeyUsage=critical,keyCertSign\n' > "$ext" ;;
    notca)
      printf 'basicConstraints=CA:FALSE\nkeyUsage=digitalSignature\n' > "$ext" ;;
  esac
  if [ "$kind" = "leaf_v15" ]; then so=$(sigopts v15); else so=$(sigopts "${sigkt:-$ikt}"); fi
  serial=$((serial + 1))
  # shellcheck disable=SC2046
  $OPENSSL x509 -req -in "$work/$name.csr" \
    -CA "$work/$issuer.pem" -CAkey "$work/$issuer.key" -set_serial $serial \
    -days $days $so -extfile "$ext" -out "$work/$name.pem" 2>/dev/null
  $OPENSSL x509 -in "$work/$name.pem" -outform DER -out "$name.der"
}

# Flip the final byte of a DER file (the tail of the signature BIT
# STRING) to corrupt the outer signature without breaking DER framing.
tamper() { # tamper <in.der> <out.der>
  python3 - "$1" "$2" <<'EOF'
import sys
data = bytearray(open(sys.argv[1], 'rb').read())
data[-1] ^= 0xff
open(sys.argv[2], 'wb').write(data)
EOF
}

echo "== good chains =="
mkroot g01_root ec
mkcert g01_leaf ec g01_root ec leaf "DNS:g01.corpus.test"

mkroot g02_root ec
mkcert g02_int ec g02_root ec ca
mkcert g02_leaf ec g02_int ec leaf "DNS:g02.corpus.test"

mkroot g03_root rsa2048
mkcert g03_leaf rsa2048 g03_root rsa2048 leaf "DNS:g03.corpus.test"

mkroot g04_root rsa2048
mkcert g04_int rsa2048 g04_root rsa2048 ca
mkcert g04_leaf rsa2048 g04_int rsa2048 leaf "DNS:g04.corpus.test"

mkroot g05_root rsa2048                      # mixed: ECDSA leaf under RSA path
mkcert g05_int rsa2048 g05_root rsa2048 ca
mkcert g05_leaf ec g05_int rsa2048 leaf "DNS:g05.corpus.test"

mkroot g06_root ec                           # mixed: RSA leaf under ECDSA path
mkcert g06_int ec g06_root ec ca
mkcert g06_leaf rsa2048 g06_int ec leaf "DNS:g06.corpus.test"

mkroot g07_root ec                           # IP SAN
mkcert g07_leaf ec g07_root ec leaf "IP:203.0.113.7"

mkroot g08_root rsa3072
mkcert g08_leaf rsa3072 g08_root rsa3072 leaf "DNS:g08.corpus.test"

mkroot g09_root rsa4096
mkcert g09_leaf rsa4096 g09_root rsa4096 leaf "DNS:g09.corpus.test"

mkroot g10_root ec                           # wildcard SAN
mkcert g10_leaf ec g10_root ec leaf "DNS:*.g10.corpus.test"

mkroot g11_root ed25519                      # real-openssl Ed25519
mkcert g11_leaf ed25519 g11_root ed25519 leaf "DNS:g11.corpus.test"

mkroot g12_root ec                           # positive pin case
mkcert g12_leaf ec g12_root ec leaf "DNS:g12.corpus.test"

# g13: WebPKI intermediate-as-anchor shape (e.g. GTS WE1). A P-384
# external root signs the P-256 intermediate with ecdsa-with-SHA384;
# only the intermediate enters the trust store, so its own signature
# is never verified — but it must PARSE (SignatureAlgorithm::
# Unsupported), and the ECDSA-SHA256 leaf under it must verify.
mkroot g13_extroot ec384
mkcert g13_ca ec g13_extroot ec384 ca
mkcert g13_leaf ec g13_ca ec leaf "DNS:g13.corpus.test"
rm g13_extroot.der                           # external root not part of the corpus

echo "== bad chains =="
mkroot b01_root ec                           # tampered ECDSA leaf sig
mkcert b01_leaf_ok ec b01_root ec leaf "DNS:b01.corpus.test"
tamper b01_leaf_ok.der b01_leaf.der && rm b01_leaf_ok.der

mkroot b02_root rsa2048                      # tampered RSA leaf sig
mkcert b02_leaf_ok rsa2048 b02_root rsa2048 leaf "DNS:b02.corpus.test"
tamper b02_leaf_ok.der b02_leaf.der && rm b02_leaf_ok.der

mkroot b03_root ec                           # unknown anchor: store gets b03_other
mkroot b03_other ec
mkcert b03_leaf ec b03_root ec leaf "DNS:b03.corpus.test"

mkroot b04_root rsa2048                      # PKCS#1 v1.5 leaf (policy refusal)
mkcert b04_leaf rsa2048 b04_root rsa2048 leaf_v15 "DNS:b04.corpus.test"

mkroot b05_root ec                           # hostname mismatch (test asks b05-wrong)
mkcert b05_leaf ec b05_root ec leaf "DNS:b05.corpus.test"

mkroot b06_root ec                           # expired: 1-day leaf vs 2026-09-22 now
mkcert b06_leaf ec b06_root ec leaf_short "DNS:b06.corpus.test"

mkroot b07_root ec                           # non-CA intermediate
mkcert b07_notca ec b07_root ec notca
mkcert b07_leaf ec b07_notca ec leaf "DNS:b07.corpus.test"

mkroot b08_root ec                           # leaf without SAN
mkcert b08_leaf ec b08_root ec leaf_nosan

mkroot b09_root ec                           # leaf without serverAuth EKU
mkcert b09_leaf ec b09_root ec leaf_noeku "DNS:b09.corpus.test"

mkroot b10_root ec                           # wrong pin (test pins b10_other's print)
mkroot b10_other ec
mkcert b10_leaf ec b10_root ec leaf "DNS:b10.corpus.test"

# b11: SHA-384-signed chain link. Parses (Unsupported) but a link
# whose signature actually needs verifying must refuse — P-384/SHA-384
# verification is not implemented.
mkroot b11_root ec
mkcert b11_leaf ec b11_root ec leaf "DNS:b11.corpus.test" ec384

echo "== done: $(ls -1 *.der | wc -l | tr -d ' ') DER files =="
