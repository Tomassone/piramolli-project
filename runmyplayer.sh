#!/bin/bash

# ==============================================================================
# Tablut Challenge - Script di Avvio Player
# Questo script fa da tramite tra l'ambiente di gara e il player Python.
# ==============================================================================

if [ "$#" -ne 3 ]; then
    echo "Errore: numero di parametri errato."
    echo "Uso: ./runmyplayer.sh <ROLE> <TIMEOUT> <SERVER_IP>"
    echo "Esempio: ./runmyplayer.sh WHITE 60 192.168.20.101"
    exit 1
fi

ROLE=$(echo "$1" | tr '[:lower:]' '[:upper:]') #case insensitivity
TIMEOUT=$2
SERVER_IP=$3

if [ "$ROLE" != "WHITE" ] && [ "$ROLE" != "BLACK" ]; then
    echo "Errore: Il ruolo deve essere WHITE o BLACK (ricevuto: $1)"
    exit 1
fi

echo "Avvio player AI: Ruolo=$ROLE | Timeout=${TIMEOUT}s | Server=$SERVER_IP"

exec python3 player.py "$ROLE" "$TIMEOUT" "$SERVER_IP"