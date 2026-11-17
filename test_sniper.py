"""Suíte de Testes Unitários para o Motor MicroSniper (Regras 1 a 5)."""

import time
from micro_sniper import MicroSniper, PreDecisaoSniper


def test_micro_sniper_call_trigger():
    """Testa fluxo de CALL: retrai até máxima anterior e reverte no 1º tick de alta."""
    sniper = MicroSniper(ml_threshold=0.55)
    vela_ant = {"open": 1.1000, "high": 1.1020, "low": 1.0990, "close": 1.1015}
    alvo = sniper.registrar_pre_decisao(
        par="EURUSD",
        timeframe="M15",
        direcao="CALL",
        vela_anterior=vela_ant,
        preco_abertura=1.1015,
        prob_win=0.62,
        lote=2.0,
        exp_minutos=15,
        duracao_vela_segundos=900,
    )

    assert alvo.taxa_alvo == 1.1020       # C1: Máxima anterior
    assert alvo.teto_invalida == 1.0990   # A: Mínima anterior
    assert alvo.max_segundos_espera == 225.0  # 1/4 de 900s

    now = time.time()
    # 1. Preço abre acima do alvo: ainda não tocou
    st, _, _ = sniper.processar_tick("EURUSD", 1.1025, ts_agora=now + 5)
    assert st == "WAITING"

    # 2. Preço recua e toca na taxa alvo (1.1018 <= 1.1020)
    st, _, _ = sniper.processar_tick("EURUSD", 1.1018, ts_agora=now + 10)
    assert st == "WAITING"
    assert alvo.taxa_atingida is True
    assert alvo.menor_preco_tocou == 1.1018

    # 3. Preço aprofunda a mínima na zona de taxa (1.1015 < 1.1018)
    st, _, _ = sniper.processar_tick("EURUSD", 1.1015, ts_agora=now + 15)
    assert st == "WAITING"
    assert alvo.menor_preco_tocou == 1.1015

    # 4. Primeiro tick de alta acima da mínima (1.1016 > 1.1015) ➔ GATILHO TIPO B!
    st, trig_alvo, motivo = sniper.processar_tick("EURUSD", 1.1016, ts_agora=now + 16)
    assert st == "TRIGGER"
    assert trig_alvo.par == "EURUSD"
    print("✅ Teste CALL Trigger aprovado:", motivo)


def test_micro_sniper_put_trigger():
    """Testa fluxo de PUT: sobe até mínima anterior e reverte no 1º tick de baixa."""
    sniper = MicroSniper(ml_threshold=0.55)
    vela_ant = {"open": 1.1050, "high": 1.1060, "low": 1.1030, "close": 1.1035}
    alvo = sniper.registrar_pre_decisao(
        par="USDJPY",
        timeframe="M5",
        direcao="PUT",
        vela_anterior=vela_ant,
        preco_abertura=1.1035,
        prob_win=0.58,
        lote=2.0,
        exp_minutos=5,
        duracao_vela_segundos=300,
    )

    assert alvo.taxa_alvo == 1.1030       # C1: Mínima anterior
    assert alvo.teto_invalida == 1.1060   # A: Máxima anterior
    assert alvo.max_segundos_espera == 75.0  # 1/4 de 300s

    now = time.time()
    # 1. Preço sobe e toca na taxa alvo de PUT (1.1032 >= 1.1030)
    st, _, _ = sniper.processar_tick("USDJPY", 1.1032, ts_agora=now + 10)
    assert st == "WAITING"
    assert alvo.taxa_atingida is True

    # 2. Primeiro tick de baixa abaixo da máxima atingida (1.1031 < 1.1032) ➔ GATILHO!
    st, trig_alvo, motivo = sniper.processar_tick("USDJPY", 1.1031, ts_agora=now + 11)
    assert st == "TRIGGER"
    print("✅ Teste PUT Trigger aprovado:", motivo)


def test_micro_sniper_invalidation_assassin_candle():
    """Testa Regra 2: Violação do teto estrutural (Vela Assassina) aborta na hora."""
    sniper = MicroSniper(ml_threshold=0.55)
    vela_ant = {"open": 1.1000, "high": 1.1020, "low": 1.0990, "close": 1.1015}
    sniper.registrar_pre_decisao(
        par="EURJPY",
        timeframe="M15",
        direcao="CALL",
        vela_anterior=vela_ant,
        preco_abertura=1.1015,
        prob_win=0.60,
        lote=1.0,
        exp_minutos=15,
    )

    now = time.time()
    # Preço derrete violentamente abaixo da mínima anterior (1.0985 < 1.0990)
    st, _, motivo = sniper.processar_tick("EURJPY", 1.0985, ts_agora=now + 20)
    assert st == "INVALIDATED"
    assert "Vela assassina" in motivo
    print("✅ Teste Invalidação Estrutural aprovado:", motivo)


def test_micro_sniper_timeout_quarter_rule():
    """Testa Regra 5: Limite de 1/4 do tempo da vela sem atingir a taxa cancela."""
    sniper = MicroSniper(ml_threshold=0.55)
    vela_ant = {"open": 1.1000, "high": 1.1020, "low": 1.0990, "close": 1.1015}
    sniper.registrar_pre_decisao(
        par="AUDUSD",
        timeframe="M5",
        direcao="CALL",
        vela_anterior=vela_ant,
        preco_abertura=1.1015,
        prob_win=0.60,
        lote=1.0,
        exp_minutos=5,
        duracao_vela_segundos=300,
    )

    now = time.time()
    # 76 segundos decorridos (maior que 75s = 1/4 de 300s)
    st, _, motivo = sniper.processar_tick("AUDUSD", 1.1030, ts_agora=now + 76)
    assert st == "TIMEOUT"
    assert "Timeout de 1/4" in motivo
    print("✅ Teste Timeout de 1/4 aprovado:", motivo)


if __name__ == "__main__":
    test_micro_sniper_call_trigger()
    test_micro_sniper_put_trigger()
    test_micro_sniper_invalidation_assassin_candle()
    test_micro_sniper_timeout_quarter_rule()
    print("\n🎉 TODOS OS TESTES DO MICRO-SNIPER FORAM APROVADOS COM 100% DE SUCESSO!")
