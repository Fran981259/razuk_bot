-- ==============================================================================
-- Indicador: EMA Wave + Clock Filter (Raghee Horner) para IQ Option
-- Linguagem: Lua (Quadcode Script / IQ Option)
-- ==============================================================================

instrument { name = "EMA Wave Clock (Raghee Horner)", overlay = true }

-- Parâmetros da Onda
period     = input(34, "Período da Onda (EMA)", input.integer, 5, 200)
slope_bars = input(3,  "Barras de Inclinação", input.integer, 1, 10)

-- Cálculo das 3 Médias Exponenciais da Onda (Fibonacci 34)
wave_high  = ema(high, period)
wave_close = ema(close, period)
wave_low   = ema(low, period)

-- Cálculo da Inclinação Normalizada
delta_ema  = wave_close - wave_close[slope_bars]
tr_range   = max(high - low, 0.00001)
slope_norm = (delta_ema / (tr_range * slope_bars)) * 2.5

-- Classificação do Relógio:
-- Subida (Alta Forte): slope_norm > 0.5
-- Descida (Baixa Forte): slope_norm < -0.7
-- Lateral (Mercado Consolidado): entre -0.7 e 0.5
is_bull = slope_norm > 0.5
is_bear = slope_norm < -0.7

-- Definição das Cores: Linhas e Preenchimento Translúcido
line_color = "#9E9E9E"                  -- Linha Cinza (Lateral)
fill_color = rgba(158, 158, 158, 0.22)  -- Preenchimento Cinza transparente

if is_bull then
    line_color = "#00E676"              -- Linha Verde (Subida)
    fill_color = rgba(0, 230, 118, 0.28)-- Preenchimento Verde transparente
elseif is_bear then
    line_color = "#FF1744"              -- Linha Vermelha (Descida)
    fill_color = rgba(255, 23, 68, 0.28)-- Preenchimento Vermelho transparente
end

-- 1. Plotagem das Linhas com espessura visível
plot(wave_high,  "Wave High (Topo)",   line_color, 2)
plot(wave_close, "Wave Close (Centro)", line_color, 3)
plot(wave_low,   "Wave Low (Fundo)",   line_color, 2)

-- 2. Preenchimento Colorido Translúcido entre as bandas
fill(wave_high, wave_low, "Preenchimento EMA Wave", fill_color)

-- 3. Sinais de Entrada e Leitura do Relógio de Raghee Horner
is_candle_bull = close > open
is_candle_bear = close < open

-- Disparos de CALL e PUT com indicação do Relógio
call_signal = is_bull and (low <= wave_high) and (close >= wave_low) and is_candle_bull
put_signal  = is_bear and (high >= wave_low) and (close <= wave_high) and is_candle_bear

plot_shape(call_signal, "Sinal CALL", shape_style.triangleup,   shape_size.normal, "#00E676", shape_location.belowbar, 0, "CALL 12h-2h", "#00E676")
plot_shape(put_signal,  "Sinal PUT",  shape_style.triangledown, shape_size.normal, "#FF1744", shape_location.abovebar, 0, "PUT 3h30-5h", "#FF1744")

-- 4. Marcadores de Transição de Horário do Relógio
-- Avisa no gráfico o exato momento em que o relógio virou para Alta, Baixa ou Lateral
became_bull = is_bull and not is_bull[1]
became_bear = is_bear and not is_bear[1]
became_side = not is_bull and not is_bear and (is_bull[1] or is_bear[1])

plot_shape(became_bull, "Relogio Alta",    shape_style.circle, shape_size.small, "#00E676", shape_location.belowbar, 0, "🕐 12h-2h", "#00E676")
plot_shape(became_bear, "Relogio Baixa",   shape_style.circle, shape_size.small, "#FF1744", shape_location.abovebar, 0, "🕔 3h30-5h", "#FF1744")
plot_shape(became_side, "Relogio Lateral", shape_style.circle, shape_size.tiny,  "#9E9E9E", shape_location.abovebar, 0, "🕒 3h (Neutro)", "#BDBDBD")

