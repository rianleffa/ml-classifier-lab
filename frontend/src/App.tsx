import { FormEvent, useEffect, useMemo, useState } from 'react'

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

type Dataset = {
  name: string; source: string; records: number; features_count: number; classes_count: number
  class_distribution: Record<string, number>; classes: { label: number; name: string }[]
  features: { name: string; description: string }[]; split: { train_percent: number; test_percent: number; stratified: boolean; note: string }
}
type Run = {
  id: string; created_at: string; metrics: Record<string, number>; confusion_matrix: number[][]
  confusion_labels: string[]; classification_report: { class: string; precision: number; recall: number; f1: number; support: number }[]
  cv_scores: number[]; class_summary: { label: number; name: string; support: number; correct: number; incorrect: number }[]
  counts: { total: number; train: number; test: number; correct: number; incorrect: number; classes: Record<string, number> }
  config: { algorithm: string; max_depth: number; test_size: number; train_size: number; random_state: number; stratified: boolean; cross_validation: string; positive_class: string }
  elapsed_seconds: number; charts: Record<string, string>
}
type SampleData = { columns: string[]; rows: Record<string, string | number>[] }
type Page = 'dashboard' | 'problem' | 'dataset' | 'training' | 'evaluation' | 'analysis' | 'conclusion' | 'code' | 'reports'

const navigation: { id: Page; title: string; glyph: string; group: string }[] = [
  { id: 'dashboard', title: 'Visão geral da AP2', glyph: '⌂', group: 'PROJETO' },
  { id: 'problem', title: '1. Problema e fluxo', glyph: '◈', group: 'RELATÓRIO AP2' },
  { id: 'dataset', title: '2. Aprendizado e dataset', glyph: '▤', group: 'RELATÓRIO AP2' },
  { id: 'training', title: '4. Algoritmo e treino', glyph: '⟳', group: 'RELATÓRIO AP2' },
  { id: 'evaluation', title: '5. Avaliação aprofundada', glyph: '◫', group: 'RELATÓRIO AP2' },
  { id: 'analysis', title: '6. Análise crítica', glyph: '⌁', group: 'RELATÓRIO AP2' },
  { id: 'conclusion', title: '7. Conclusão', glyph: '✓', group: 'RELATÓRIO AP2' },
  { id: 'code', title: '8. Código e execução', glyph: '</>', group: 'ANEXOS E EVIDÊNCIAS' },
  { id: 'reports', title: 'Relatórios e evidências', glyph: '⇩', group: 'ANEXOS E EVIDÊNCIAS' },
]
const pageInfo: Record<Page, { title: string; subtitle: string }> = {
  dashboard: { title: 'Laboratório de Machine Learning', subtitle: 'Acompanhe o fluxo completo da análise, dos dados às métricas.' },
  problem: { title: '1. Introdução e definição do problema', subtitle: 'Tema, objetivo e fluxo de Machine Learning descritos no relatório.' },
  dataset: { title: '2. Tipo de aprendizado e dataset', subtitle: 'Dados oficiais carregados diretamente do Scikit-learn.' },
  training: { title: '4. Algoritmo clássico: implementação e resultados', subtitle: 'Configure a árvore e execute o treinamento real.' },
  evaluation: { title: '5. Avaliação aprofundada do modelo', subtitle: 'Métricas calculadas sobre previsões e validação cruzada.' },
  analysis: { title: '6. Análise crítica do desempenho', subtitle: 'Interpretação baseada nos resultados desta execução.' },
  conclusion: { title: '7. Conclusão', subtitle: 'Síntese da avaliação atual e dos limites deste estudo acadêmico.' },
  code: { title: '8. Anexo: código-fonte e execução', subtitle: 'Fonte exata da rotina de treinamento executada pela API.' },
  reports: { title: 'Relatórios e evidências', subtitle: 'Histórico de execuções e arquivos para a entrega da AP2.' },
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, options)
  if (!response.ok) {
    let message = `A API respondeu com status ${response.status}.`
    try { const body = await response.json(); message = body.detail ?? message } catch { /* resposta sem JSON */ }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

function fmt(value?: number, digits = 3) { return typeof value === 'number' ? value.toFixed(digits) : '—' }
function dateFmt(value: string) { return new Intl.DateTimeFormat('pt-BR', { dateStyle: 'short', timeStyle: 'short' }).format(new Date(value)) }
function download(path: string) { window.open(`${API}${path}`, '_blank', 'noopener,noreferrer') }
function escapeHtml(value: string) { return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;') }

function syntaxHighlight(source: string) {
  const pattern = /(#[^\n]*|"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:from|import|as|def|return|if|else|for|in|try|except|with|and|or|not|is|None|True|False|class)\b|\b\d+(?:\.\d+)?\b)/g
  let html = ''; let last = 0
  for (const match of source.matchAll(pattern)) {
    const index = match.index ?? 0; const token = match[0]
    html += escapeHtml(source.slice(last, index))
    const kind = token.startsWith('#') ? 'comment' : token.startsWith('"') || token.startsWith("'") ? 'string' : /^\d/.test(token) ? 'number' : 'keyword'
    html += `<span class="tok-${kind}">${escapeHtml(token)}</span>`; last = index + token.length
  }
  return html + escapeHtml(source.slice(last))
}

export default function App() {
  const [page, setPage] = useState<Page>('dashboard')
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [samples, setSamples] = useState<SampleData | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [run, setRun] = useState<Run | null>(null)
  const [source, setSource] = useState('')
  const [maxDepth, setMaxDepth] = useState(4)
  const [testSize, setTestSize] = useState(0.2)
  const [randomState, setRandomState] = useState(42)
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)
  const [mobileMenu, setMobileMenu] = useState(false)
  const [toast, setToast] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    Promise.all([request<Dataset>('/api/dataset'), request<SampleData>('/api/dataset/samples'), request<{ runs: Run[] }>('/api/runs')])
      .then(([d, s, r]) => { if (!active) return; setDataset(d); setSamples(s); setRuns(r.runs); setRun(r.runs[0] ?? null) })
      .catch(e => { if (active) setError(e instanceof Error ? e.message : 'Falha ao conectar com a API.') })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (!toast) return
    const timer = window.setTimeout(() => setToast(''), 3600)
    return () => window.clearTimeout(timer)
  }, [toast])

  const navigate = (target: Page) => { setPage(target); setMobileMenu(false); setError('') }
  const executeTraining = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError('')
    try {
      const result = await request<Run>('/api/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ max_depth: maxDepth, test_size: testSize, random_state: randomState }) })
      setRun(result); setRuns(current => [result, ...current]); setToast('Treinamento concluído. Métricas e gráficos atualizados.'); setPage('evaluation')
    } catch (e) { setError(e instanceof Error ? e.message : 'Não foi possível treinar o modelo.') }
    finally { setBusy(false) }
  }
  const copyCode = async () => {
    try { await navigator.clipboard.writeText(source); setToast('Código copiado para a área de transferência.') }
    catch { setError('Não foi possível acessar a área de transferência neste navegador.') }
  }
  const loadCode = async () => {
    try { const data = await request<{ source: string }>('/api/code'); setSource(data.source) }
    catch (e) { setError(e instanceof Error ? e.message : 'Falha ao carregar o código.') }
  }
  useEffect(() => { if (page === 'code' && !source) void loadCode() }, [page]) // source is loaded from the executed backend module

  const classRatio = useMemo(() => {
    if (!dataset) return null
    const values = Object.values(dataset.class_distribution); return Math.max(...values) / Math.min(...values)
  }, [dataset])

  return <div className="app-shell">
    <aside className={`sidebar ${mobileMenu ? 'sidebar-open' : ''}`}>
      <div className="brand"><div className="brand-mark"><span>ML</span><i /></div><div><strong>ML Classifier <b>Lab</b></strong><small>ULBRA · PROJETO AP2</small></div></div>
      <div className="workspace-label">ESPAÇO DE TRABALHO</div>
      <nav>{navigation.map((item, index) => <div key={item.id}>{(index === 0 || navigation[index - 1].group !== item.group) && <div className="nav-group">{item.group}</div>}<button className={`nav-item ${page === item.id ? 'active' : ''}`} onClick={() => navigate(item.id)}><span className="nav-glyph">{item.glyph}</span>{item.title}{page === item.id && <span className="active-dot" />}</button></div>)}</nav>
      <div className="sidebar-bottom"><div className="api-indicator"><span className={error && !dataset ? 'dot-red' : 'dot-green'} />{loading ? 'Conectando à API…' : error && !dataset ? 'API indisponível' : 'API conectada'}</div><div className="academic-note">Projeto acadêmico<br />Sem finalidade diagnóstica.</div></div>
    </aside>
    <div className="content-shell">
      <header className="topbar"><button className="mobile-menu-button" onClick={() => setMobileMenu(v => !v)} aria-label="Abrir menu">☰</button><div className="breadcrumbs"><span>ML Classifier Lab</span><i>/</i><strong>{pageInfo[page].title}</strong></div><div className="topbar-right"><span className="status-chip"><i /> Ambiente acadêmico</span><div className="avatar">AP</div></div></header>
      <main className="main-content"><div className="page-heading"><div><div className="eyebrow">FUNDAMENTOS DE INTELIGÊNCIA ARTIFICIAL</div><h1>{pageInfo[page].title}</h1><p>{pageInfo[page].subtitle}</p></div>{page === 'dashboard' && <button className="button primary" onClick={() => navigate('training')}><span>＋</span> Nova execução</button>}</div>
        {error && <div className="alert error-alert"><span>!</span><div><strong>Não foi possível concluir a operação</strong><p>{error}</p></div><button onClick={() => setError('')} aria-label="Fechar">×</button></div>}
        {loading ? <Loading /> : <>
          {page === 'dashboard' && <Dashboard dataset={dataset} run={run} runs={runs} navigate={navigate} />}
          {page === 'problem' && <Problem />}
          {page === 'dataset' && <DatasetPage dataset={dataset} samples={samples} />}
          {page === 'training' && <TrainingPage maxDepth={maxDepth} setMaxDepth={setMaxDepth} testSize={testSize} setTestSize={setTestSize} randomState={randomState} setRandomState={setRandomState} busy={busy} onSubmit={executeTraining} />}
          {page === 'evaluation' && <Evaluation run={run} />}
          {page === 'analysis' && <Analysis run={run} dataset={dataset} classRatio={classRatio} />}
          {page === 'conclusion' && <Conclusion run={run} />}
          {page === 'code' && <CodePage source={source} copyCode={copyCode} />}
          {page === 'reports' && <Reports runs={runs} run={run} selectRun={setRun} />}
        </>}
      </main>
      <footer className="footer"><span>ML Classifier Lab <b>·</b> ULBRA Torres · Rian Cardoso Leffa</span><span>Scikit-learn · FastAPI · React</span></footer>
    </div>
    {mobileMenu && <button className="mobile-backdrop" onClick={() => setMobileMenu(false)} aria-label="Fechar menu" />}
    {toast && <div className="toast"><span>✓</span>{toast}</div>}
  </div>
}

function Loading() { return <div className="loading-card"><div className="spinner" /><strong>Carregando dados do laboratório</strong><span>Consultando o backend e o dataset do Scikit-learn…</span></div> }
function Empty({ title = 'Nenhuma execução disponível', text = 'Inicie um treinamento para calcular métricas reais e gerar os gráficos desta análise.' }: { title?: string; text?: string }) { return <div className="empty-state"><div className="empty-icon">◷</div><strong>{title}</strong><p>{text}</p></div> }
function MetricCard({ label, value, detail, icon, tone = 'blue' }: { label: string; value: string; detail: string; icon: string; tone?: string }) { return <div className="metric-card"><div className={`metric-icon ${tone}`}>{icon}</div><div className="metric-label">{label}</div><div className="metric-value">{value}</div><div className="metric-detail">{detail}</div></div> }
function SectionTitle({ title, detail, action }: { title: string; detail?: string; action?: React.ReactNode }) { return <div className="section-title"><div><h2>{title}</h2>{detail && <p>{detail}</p>}</div>{action}</div> }

function Dashboard({ dataset, run, runs, navigate }: { dataset: Dataset | null; run: Run | null; runs: Run[]; navigate: (p: Page) => void }) {
  return <>
    <div className="hero-banner"><div className="hero-copy"><span className="hero-tag"><i /> APRENDIZADO DE MÁQUINA · RELATÓRIO AP2</span><h2>Classificação de tumores mamários utilizando <em>Árvore de Decisão</em></h2><p>Fundamentos de Inteligência Artificial · Análise e Desenvolvimento de Sistemas · ULBRA Torres</p><button className="button primary" onClick={() => navigate(run ? 'evaluation' : 'training')}>{run ? 'Ver última avaliação' : 'Começar primeira análise'} <span>→</span></button></div><div className="hero-graphic"><div className="report-cover-card"><small>RELATÓRIO DA AVALIAÇÃO PARCIAL 2 · AP2</small><b>Rian Cardoso Leffa</b><span>Fundamentos de Inteligência Artificial</span><span>Análise e Desenvolvimento de Sistemas · ULBRA Torres</span><i>17 · 10 · 2026</i></div></div></div>
    <div className="metric-grid">
      <MetricCard label="Dataset carregado" value={dataset ? 'Disponível' : 'Indisponível'} detail={dataset?.name ?? 'Verifique a API'} icon="▤" tone="blue" />
      <MetricCard label="Registros · atributos" value={dataset ? `${dataset.records} · ${dataset.features_count}` : '—'} detail="Breast Cancer Wisconsin" icon="▦" tone="violet" />
      <MetricCard label="Algoritmo selecionado" value="Árvore de decisão" detail={run ? `Profundidade máxima ${run.config.max_depth}` : 'DecisionTreeClassifier'} icon="⌘" tone="green" />
      <MetricCard label="Acurácia do modelo" value={run ? `${fmt(run.metrics.accuracy * 100, 1)}%` : '—'} detail={run ? 'Último conjunto de teste' : 'Treine para calcular'} icon="◉" tone="amber" />
    </div>
    <div className="dashboard-lower"><section className="panel chart-panel"><SectionTitle title="Resumo da avaliação" detail={run ? `Execução ${run.id.slice(0, 8)} · ${dateFmt(run.created_at)}` : 'Métricas calculadas após o treinamento'} action={run && <button className="text-action" onClick={() => navigate('evaluation')}>Ver avaliação →</button>} />{run ? <div className="dashboard-metrics"><div className="bars-area">{[['Precisão', run.metrics.precision], ['Recall', run.metrics.recall], ['F1-score', run.metrics.f1], ['CV média', run.metrics.cv_mean]].map(([name, score]) => <div className="horizontal-bar" key={String(name)}><span>{name}</span><div><i style={{ width: `${Number(score) * 100}%` }} /></div><b>{fmt(Number(score))}</b></div>)}</div><div className="mini-confusion"><span>REAL ↓ · PREVISTO →</span><div className="matrix"><b></b><b>Maligno</b><b>Benigno</b><b>Maligno</b><b>{run.confusion_matrix[0][0]}</b><b className="cell-danger">{run.confusion_matrix[0][1]}</b><b>Benigno</b><b className="cell-warn">{run.confusion_matrix[1][0]}</b><b>{run.confusion_matrix[1][1]}</b></div><small>Falso negativo: maligno previsto como benigno.</small></div></div> : <Empty />}</section>
      <section className="panel recent-panel"><SectionTitle title="Execuções recentes" action={<button className="icon-link" onClick={() => navigate('reports')} aria-label="Abrir relatórios">↗</button>} />{runs.length ? <div className="recent-list">{runs.slice(0, 4).map((item, i) => <button className="recent-row" key={item.id} onClick={() => navigate('reports')}><span className={`run-icon r${i % 3}`}>◈</span><span className="recent-info"><b>Árvore · profundidade {item.config.max_depth}</b><small>{dateFmt(item.created_at)}</small></span><strong>{fmt(item.metrics.accuracy * 100, 1)}%</strong><i>›</i></button>)}</div> : <Empty title="Sem execuções ainda" text="Seus treinamentos concluídos aparecerão aqui." />}<button className="button outline full" onClick={() => navigate('training')}>＋ Iniciar treinamento</button></section></div>
    <div className="notice-strip"><span>ⓘ</span><p><b>Uso acadêmico:</b> este projeto demonstra técnicas de Machine Learning e não deve ser utilizado para diagnóstico ou decisão clínica.</p><button onClick={() => navigate('problem')}>Sobre o estudo de caso →</button></div>
  </>
}

function Problem() {
  const steps = [['01', 'Definição do problema', 'Classificar tumores em duas categorias a partir de medidas celulares.'], ['02', 'Carregamento dos dados', 'Usar o Breast Cancer Wisconsin Diagnostic do Scikit-learn.'], ['03', 'Preparação dos dados', 'Separar treino e teste de forma estratificada e reprodutível.'], ['04', 'Treinamento', 'Ajustar uma Árvore de Decisão sobre os exemplos rotulados.'], ['05', 'Avaliação', 'Medir previsões no teste e usar validação cruzada estratificada.'], ['06', 'Implantação e monitoramento', 'Em uma aplicação real, validar externamente e monitorar deriva.']]
  return <><div className="problem-intro"><div className="intro-symbol">◈</div><div><span className="pill blue-pill">ESTUDO DE CASO</span><h2>Classificação de tumores mamários<br />utilizando Árvore de Decisão</h2><p>O objetivo é estimar a classe de um registro a partir de características numéricas observadas em imagens digitalizadas de amostras celulares.</p></div></div><div className="info-grid"><div className="panel info-card"><span className="info-icon blue">◎</span><h3>Por que esse problema?</h3><p>A classificação binária permite estudar como um algoritmo aprende relações entre medidas de entrada e rótulos conhecidos. O contexto também evidencia por que erros de classificação precisam ser analisados por classe.</p></div><div className="panel info-card"><span className="info-icon green">✳</span><h3>Aprendizado supervisionado</h3><p>O dataset inclui exemplos com atributos e diagnósticos já rotulados. O modelo aprende com o conjunto de treinamento e é avaliado em exemplos separados, cujos rótulos são conhecidos.</p></div></div><div className="panel variables-panel"><SectionTitle title="Entradas e alvo" detail="O dataset disponibiliza 30 atributos numéricos por registro."/><div className="variable-columns"><div><span className="pill gray-pill">VARIÁVEIS DE ENTRADA · X</span><h3>Medidas do núcleo celular</h3><p>Raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal. Para cada medida existem estatísticas de média, erro padrão e pior caso.</p></div><div className="target-box"><span className="pill green-pill">ALVO · y</span><h3>Diagnóstico rotulado</h3><p><b>Maligno</b> (classe 0) ou <b>Benigno</b> (classe 1), conforme os rótulos do dataset.</p></div></div></div><div className="panel flow-panel"><SectionTitle title="Fluxo do projeto de Machine Learning" detail="Da formulação ao monitoramento de uma eventual implantação."/><div className="flow-list">{steps.map(([n, title, text], i) => <div className="flow-step" key={n}><div className={`step-number ${i === 4 ? 'current' : ''}`}>{n}</div><div><b>{title}</b><p>{text}</p></div>{i < steps.length - 1 && <span className="flow-arrow">→</span>}</div>)}</div></div><div className="medical-warning"><span>⚕</span><div><b>Finalidade acadêmica</b><p>Os dados, o modelo e as métricas desta demonstração não representam validação clínica. Não utilizar para diagnóstico médico.</p></div></div></>
}

function DatasetPage({ dataset, samples }: { dataset: Dataset | null; samples: SampleData | null }) {
  if (!dataset) return <Empty title="Dataset indisponível" text="A API não conseguiu carregar o pacote de dados do Scikit-learn." />
  const total = Object.values(dataset.class_distribution).reduce((a, b) => a + b, 0)
  return <><div className="metric-grid dataset-stats"><MetricCard label="Registros" value={String(dataset.records)} detail="Instâncias do dataset" icon="▤" /><MetricCard label="Atributos" value={String(dataset.features_count)} detail="Variáveis numéricas de entrada" icon="⌘" tone="violet" /><MetricCard label="Classes" value={String(dataset.classes_count)} detail="Maligno e benigno" icon="◉" tone="green" /><MetricCard label="Divisão padrão" value={`${dataset.split.train_percent} / ${dataset.split.test_percent}`} detail="Treino / teste · estratificada" icon="⇄" tone="amber" /></div><div className="panel learning-type-panel"><span className="pill blue-pill">TIPO DE APRENDIZADO</span><h2>Aprendizado supervisionado · classificação binária</h2><p>Cada um dos {dataset.records} registros tem 30 atributos numéricos e um rótulo conhecido: <b>Maligno (0)</b> ou <b>Benigno (1)</b>. O algoritmo aprende com esses exemplos rotulados.</p></div><div className="dataset-grid"><section className="panel"><SectionTitle title="Distribuição por classe" detail="Contagens reais no conjunto completo"/><div className="distribution-rows">{Object.entries(dataset.class_distribution).map(([name, count], i) => <div className="distribution-row" key={name}><div className="distribution-label"><span className={`class-dot d${i}`} /><b>{name}</b><span>{count} registros</span><strong>{(count / total * 100).toFixed(1)}%</strong></div><div className="distribution-track"><i className={`distribution-fill f${i}`} style={{ width: `${count / total * 100}%` }} /></div></div>)}</div><div className="source-note">Fonte: <code>{dataset.source}</code></div><button className="button outline" onClick={() => download('/api/dataset.csv')}>⇩ Baixar dataset completo em CSV</button></section><section className="panel"><SectionTitle title="Sobre a divisão"/><div className="split-visual"><div><b>{dataset.split.train_percent}%</b><span>Treinamento</span></div><i /><div><b>{dataset.split.test_percent}%</b><span>Teste</span></div></div><p className="body-copy">A separação mantém a proporção das classes nos dois subconjuntos. Na página de treinamento você pode ajustar a proporção de teste; a divisão será refeita para cada execução.</p><div className="small-callout">Mesmo dataset e mapeamento de classes são usados na exploração e no treinamento.</div></section></div><section className="panel table-panel"><SectionTitle title="Amostras dos dados" detail={`${samples?.rows.length ?? 0} primeiras linhas do conjunto real`} action={<button className="text-action" onClick={() => download('/api/dataset.csv')}>Baixar CSV completo ↓</button>} />{samples ? <div className="table-scroll"><table><thead><tr>{['#', ...samples.columns.slice(0, 7), '…', 'diagnostico'].map((c, i) => <th key={`${c}-${i}`}>{c}</th>)}</tr></thead><tbody>{samples.rows.map((row, i) => <tr key={i}><td>{i + 1}</td>{samples.columns.slice(0, 7).map(c => <td key={c}>{typeof row[c] === 'number' ? (row[c] as number).toFixed(3) : row[c]}</td>)}<td>…</td><td><span className={`class-badge ${row.diagnostico === 'Maligno' ? 'malignant' : 'benign'}`}>{row.diagnostico}</span></td></tr>)}</tbody></table></div> : <Empty title="Amostras indisponíveis" />}</section><section className="panel feature-panel"><SectionTitle title="Atributos disponíveis" detail="Nome e descrição das 30 variáveis de entrada."/><div className="feature-grid">{dataset.features.map((feature, i) => <div className="feature-row" key={feature.name}><span>{String(i + 1).padStart(2, '0')}</span><b>{feature.name}</b><p>{feature.description}</p></div>)}</div></section></>
}

function TrainingPage({ maxDepth, setMaxDepth, testSize, setTestSize, randomState, setRandomState, busy, onSubmit }: { maxDepth: number; setMaxDepth: (v: number) => void; testSize: number; setTestSize: (v: number) => void; randomState: number; setRandomState: (v: number) => void; busy: boolean; onSubmit: (e: FormEvent) => void }) {
  return <div className="training-layout"><section className="panel training-form-panel"><div className="form-heading"><div className="model-icon">⌘</div><div><span className="pill blue-pill">ALGORITMO CLÁSSICO</span><h2>Árvore de Decisão</h2><p>DecisionTreeClassifier · Scikit-learn</p></div><span className="real-badge"><i /> execução real</span></div><div className="form-divider"/><form onSubmit={onSubmit}><label className="field-label" htmlFor="depth">Profundidade máxima <b>{maxDepth}</b></label><input id="depth" className="range-input" type="range" min="1" max="12" value={maxDepth} onChange={e => setMaxDepth(Number(e.target.value))}/><div className="range-caption"><span>1 · simples</span><span>12 · complexa</span></div><label className="field-label" htmlFor="test">Proporção do conjunto de teste <b>{Math.round(testSize * 100)}%</b></label><select id="test" value={testSize} onChange={e => setTestSize(Number(e.target.value))}><option value={0.2}>20% teste · 80% treino (padrão)</option><option value={0.25}>25% teste · 75% treino</option><option value={0.3}>30% teste · 70% treino</option><option value={0.1}>10% teste · 90% treino</option><option value={0.4}>40% teste · 60% treino</option></select><label className="field-label" htmlFor="seed">Estado aleatório <b>reprodutível</b></label><input id="seed" className="text-input" type="number" min="0" max="999999" value={randomState} onChange={e => setRandomState(Number(e.target.value))}/><div className="config-summary"><b>Configuração aplicada</b><div><span>Algoritmo</span><strong>DecisionTreeClassifier</strong></div><div><span>Divisão</span><strong>Estratificada · {Math.round((1 - testSize) * 100)}/{Math.round(testSize * 100)}</strong></div><div><span>Validação cruzada</span><strong>StratifiedKFold · 5 partições</strong></div></div><button className="button primary run-button" type="submit" disabled={busy}>{busy ? <><span className="button-spinner"/> Treinando modelo…</> : <>▶ Executar treinamento</>}</button><p className="form-footnote">O backend treina novamente, calcula métricas reais e gera os gráficos da execução.</p></form></section><aside className="training-side"><div className="panel info-card"><span className="info-icon blue">ⓘ</span><h3>O que acontece ao executar?</h3><ul><li>Carrega os 569 registros do Scikit-learn.</li><li>Separa treino e teste com estratificação.</li><li>Ajusta o modelo e prevê o conjunto de teste.</li><li>Calcula métricas e 5 partições de validação cruzada.</li><li>Salva os gráficos e o registro da execução.</li></ul></div><div className="medical-warning compact"><span>⚕</span><div><b>Atenção sobre o contexto</b><p>Esta ferramenta é didática e não substitui avaliação clínica.</p></div></div></aside></div>
}

function Evaluation({ run }: { run: Run | null }) {
  if (!run) return <Empty />
  const chart = (key: string, title: string, wide = false) => <div className={`panel chart-image-card ${wide ? 'wide' : ''}`}><SectionTitle title={title}/><a href={`${API}${run.charts[key]}`} target="_blank" rel="noreferrer" download><img src={`${API}${run.charts[key]}`} alt={title}/></a><a className="text-action chart-download" href={`${API}${run.charts[key]}`} download>{title} · PNG ↓</a></div>
  return <><div className="run-banner"><div><span className="pill green-pill">EXECUÇÃO CONCLUÍDA</span><h2>Resultado da Árvore de Decisão</h2><p>{dateFmt(run.created_at)} · execução <code>{run.id.slice(0, 8)}</code> · {run.elapsed_seconds.toFixed(3)} s</p></div><button className="button outline" onClick={() => download(`/api/runs/${run.id}/results.csv`)}>⇩ Exportar métricas CSV</button></div><div className="metric-grid evaluation-metrics"><MetricCard label="Acurácia" value={`${fmt(run.metrics.accuracy * 100, 1)}%`} detail={`${run.counts.correct} de ${run.counts.test} previsões corretas`} icon="◉"/><MetricCard label="Precisão · maligno" value={fmt(run.metrics.precision)} detail="Entre os previstos como malignos" icon="◎" tone="violet"/><MetricCard label="Recall · maligno" value={fmt(run.metrics.recall)} detail="Malignos identificados corretamente" icon="⌕" tone="amber"/><MetricCard label="F1-score · maligno" value={fmt(run.metrics.f1)} detail="Equilíbrio entre precisão e recall" icon="✳" tone="green"/><MetricCard label="Validação cruzada" value={`${fmt(run.metrics.cv_mean * 100, 1)}%`} detail={`Média; desvio padrão ${fmt(run.metrics.cv_std, 4)}`} icon="⟳" tone="blue"/></div><div className="chart-grid">{chart('confusion-matrix', 'Matriz de confusão')}{chart('cross-validation', 'Acurácia por partição')}{chart('class-distribution', 'Distribuição das classes no teste')}{chart('metrics', 'Comparação de métricas')}</div><section className="panel table-panel"><SectionTitle title="Relatório de classificação" detail="Precisão, recall e F1 por classe no conjunto de teste."/><div className="table-scroll"><table><thead><tr><th>Classe</th><th>Precisão</th><th>Recall</th><th>F1-score</th><th>Suporte</th></tr></thead><tbody>{run.classification_report.map(row => <tr key={row.class}><td><span className={`class-badge ${row.class === 'Maligno' ? 'malignant' : 'benign'}`}>{row.class}</span></td><td>{fmt(row.precision)}</td><td>{fmt(row.recall)}</td><td>{fmt(row.f1)}</td><td>{row.support}</td></tr>)}</tbody></table></div><div className="metric-explanations"><div><b>Acurácia</b><p>Fração de previsões corretas no total. Pode ocultar desempenho fraco em uma classe.</p></div><div><b>Precisão · maligno</b><p>Dos registros previstos como malignos, quantos eram malignos.</p></div><div><b>Recall · maligno</b><p>Dos registros realmente malignos, quantos foram identificados. O complemento corresponde à taxa de falsos negativos.</p></div><div><b>F1-score · maligno</b><p>Média harmônica de precisão e recall para a classe positiva definida como maligno.</p></div></div></section><div className="medical-warning"><span>⚕</span><div><b>Falso negativo neste contexto</b><p>Um registro maligno previsto como benigno. A matriz de confusão organiza linhas pela classe real e colunas pela classe prevista, na ordem Maligno, Benigno.</p></div></div></>
}

function Analysis({ run, dataset, classRatio }: { run: Run | null; dataset: Dataset | null; classRatio: number | null }) {
  if (!run || !dataset) return <Empty />
  const delta = run.metrics.accuracy - run.metrics.cv_mean
  const mild = classRatio !== null && classRatio < 2
  return <><div className="comparison-hero"><div><span className="pill blue-pill">LEITURA DA EXECUÇÃO</span><h2>O que os resultados mostram?</h2><p>Esta interpretação usa as métricas e contagens calculadas na execução selecionada.</p></div><div className="comparison-values"><div><small>TESTE</small><b>{fmt(run.metrics.accuracy * 100, 1)}%</b></div><span>↔</span><div><small>CV · MÉDIA</small><b>{fmt(run.metrics.cv_mean * 100, 1)}%</b></div></div></div><div className="analysis-grid"><section className="panel analysis-card"><span className="analysis-kicker">GENERALIZAÇÃO INTERNA</span><h3>{Math.abs(delta) <= 0.03 ? 'Resultados próximos' : delta > 0 ? 'Teste acima da média CV' : 'Teste abaixo da média CV'}</h3><p>Acurácia no conjunto de teste {delta >= 0 ? 'supera' : 'fica abaixo de'} a média das cinco partições em <b>{fmt(Math.abs(delta) * 100, 1)} pontos percentuais</b>. CV: {run.cv_scores.map(v => fmt(v)).join(' · ')} (média {fmt(run.metrics.cv_mean)}, desvio padrão amostral {fmt(run.metrics.cv_std)}).</p></section><section className="panel analysis-card"><span className="analysis-kicker">ERROS NO CONJUNTO DE TESTE</span><h3>{run.counts.incorrect} erro(s) em {run.counts.test} registros</h3><div className="error-class-list">{run.class_summary.map(c => <div key={c.label}><span className={`class-dot ${c.label === 0 ? 'd0' : 'd1'}`} /><b>{c.name}</b><span>{c.correct} acertos</span><strong>{c.incorrect} erros</strong></div>)}</div><p>Falsos negativos (maligno previsto benigno): <b>{run.confusion_matrix[0][1]}</b>. Falsos positivos: <b>{run.confusion_matrix[1][0]}</b>.</p></section><section className="panel analysis-card"><span className="analysis-kicker">DISTRIBUIÇÃO DAS CLASSES</span><h3>{mild ? 'Diferença observada, sem evidência de desequilíbrio grave' : 'Confira o efeito da distribuição'}</h3><p>No dataset completo, a proporção é {dataset.class_distribution.Maligno} malignos e {dataset.class_distribution.Benigno} benignos (razão entre a maior e a menor classe de {classRatio?.toFixed(2)}). A divisão estratificada preserva aproximadamente essas proporções; a métrica por classe continua necessária.</p></section><section className="panel analysis-card"><span className="analysis-kicker">LIMITAÇÕES DO MODELO</span><h3>Uma árvore simples tem limites</h3><p>A Árvore de Decisão pode variar com a amostra, criar fronteiras em degraus e sobreajustar. Profundidade limitada reduz parte desse risco, mas desempenho medido aqui não garante resultados em outra população, equipamento ou conjunto externo.</p></section></div><div className="critical-note"><b>Por que acurácia não basta?</b><p>Uma única média não mostra que tipos de erro ocorreram. Precisão e recall avaliam a classe maligna por ângulos diferentes; F1 resume esse equilíbrio. A matriz mostra contagens de acertos e confusões. Em contexto médico real, validação clínica, análise de custos dos erros e aprovação regulatória seriam indispensáveis.</p></div><p className="small-muted">Execução analisada: {run.id.slice(0, 8)} · {dateFmt(run.created_at)} · profundidade {run.config.max_depth} · teste {Math.round(run.config.test_size * 100)}%.</p></>
}

function Conclusion({ run }: { run: Run | null }) {
  return <><div className="panel conclusion-panel"><span className="pill blue-pill">SÍNTESE DO PROJETO</span><h2>Aprendizado supervisionado aplicado à classificação binária</h2><p>O projeto utilizou o Breast Cancer Wisconsin Diagnostic e uma Árvore de Decisão para estudar a classificação de registros malignos e benignos. A avaliação combina métricas no conjunto de teste com validação cruzada estratificada de cinco partições.</p>{run ? <div className="conclusion-result"><b>Resultado desta execução</b><p>Acurácia no teste de <strong>{fmt(run.metrics.accuracy * 100, 1)}%</strong>; recall da classe maligna de <strong>{fmt(run.metrics.recall * 100, 1)}%</strong>; média da validação cruzada de <strong>{fmt(run.metrics.cv_mean * 100, 1)}%</strong> (desvio-padrão {fmt(run.metrics.cv_std)}). Interprete esses valores junto com a matriz de confusão e os erros por classe.</p></div> : <div className="small-callout">Execute o modelo para incluir nesta síntese métricas calculadas. Nenhum resultado é preenchido antes de uma execução.</div>}<h3>Conclusão acadêmica</h3><p>Nenhuma métrica isolada descreve todos os aspectos de um classificador. Acurácia deve ser analisada junto com matriz de confusão, precisão, recall e F1-score. Os resultados observados descrevem este dataset e esta configuração; não garantem desempenho em outros contextos.</p><div className="medical-warning"><span>⚕</span><div><b>Limite de uso</b><p>A aplicação e o modelo têm finalidade acadêmica e não constituem uma ferramenta clínica validada. Não utilizar para diagnóstico médico.</p></div></div></div><div className="panel next-steps-panel"><SectionTitle title="Conteúdo organizado conforme o relatório"/><div className="conclusion-sections">{['1. Introdução e definição do problema', '2. Tipo de aprendizado e dataset', '3. Fluxo do projeto de Machine Learning', '4. Algoritmo clássico: implementação e resultados', '5. Avaliação aprofundada do modelo', '6. Análise crítica do desempenho', '7. Conclusão', '8. Anexo: código-fonte e execução'].map(item => <div key={item}><span>✓</span>{item}</div>)}</div></div></>
}

function CodePage({ source, copyCode }: { source: string; copyCode: () => void }) {
  if (!source) return <div className="loading-card"><div className="spinner"/><strong>Carregando fonte executada…</strong></div>
  return <><div className="code-toolbar"><div><span className="file-icon">Py</span><div><b>training.py</b><small>Fonte usada pelo endpoint de treinamento</small></div></div><div><button className="button outline" onClick={() => download('/api/code.py')}>⇩ Baixar .py</button><button className="button primary" onClick={copyCode}>▣ Copiar código</button></div></div><div className="panel code-panel"><div className="code-header"><span><i/> Python</span><span>backend/app/training.py · UTF-8</span></div><div className="code-scroll"><pre><code dangerouslySetInnerHTML={{ __html: syntaxHighlight(source) }} /></pre></div></div><div className="small-callout code-note"><b>Correspondência verificável:</b> este arquivo é importado pela API para todas as execuções. Ele carrega o dataset, treina o modelo, calcula métricas e grava os quatro gráficos exibidos na avaliação.</div></>
}

function Reports({ runs, run, selectRun }: { runs: Run[]; run: Run | null; selectRun: (r: Run) => void }) {
  return <><div className="report-intro"><div><span className="pill blue-pill">EVIDÊNCIAS DA AP2</span><h2>Resultados rastreáveis por execução</h2><p>Consulte a configuração e exporte as tabelas e gráficos que foram gerados pelo modelo.</p></div>{run && <button className="button primary" onClick={() => download(`/api/runs/${run.id}/report.pdf`)}>⇩ Gerar relatório PDF</button>}</div>{!runs.length ? <Empty title="Ainda não há relatórios" text="Execute um treinamento para salvar resultados e disponibilizar exportações."/> : <><div className="report-columns"><section className="panel runs-panel"><SectionTitle title="Histórico de execuções" detail={`${runs.length} treinamento(s) salvo(s)`}/><div className="runs-table">{runs.map((item, i) => <button className={`run-history-row ${run?.id === item.id ? 'selected' : ''}`} key={item.id} onClick={() => selectRun(item)}><span className={`run-icon r${i % 3}`}>◈</span><span className="run-history-main"><b>DecisionTreeClassifier · max_depth {item.config.max_depth}</b><small>{dateFmt(item.created_at)} · semente {item.config.random_state}</small></span><span className="run-accuracy"><b>{fmt(item.metrics.accuracy * 100, 1)}%</b><small>acurácia</small></span><span className="row-arrow">›</span></button>)}</div></section>{run && <section className="panel export-panel"><SectionTitle title="Exportar execução" detail={`Selecionada · ${run.id.slice(0, 8)}`}/><div className="export-summary"><div><span>Profundidade</span><b>{run.config.max_depth}</b></div><div><span>Divisão treino/teste</span><b>{Math.round(run.config.train_size * 100)} / {Math.round(run.config.test_size * 100)}%</b></div><div><span>Tempo de execução</span><b>{run.elapsed_seconds.toFixed(3)} s</b></div><div><span>CV média ± desvio</span><b>{fmt(run.metrics.cv_mean)} ± {fmt(run.metrics.cv_std)}</b></div></div><div className="export-actions"><button onClick={() => download(`/api/runs/${run.id}/results.csv`)}>▤ Métricas em CSV <span>↓</span></button><button onClick={() => download(`/api/runs/${run.id}/report.pdf`)}>▧ Relatório completo em PDF <span>↓</span></button>{Object.entries(run.charts).map(([name, url]) => <a key={name} href={`${API}${url}`} download>▧ Gráfico · {name.replaceAll('-', ' ')} <span>↓</span></a>)}</div></section>}</div><section className="panel ap2-checklist"><SectionTitle title="Organização sugerida para a entrega" detail="O relatório exportado reúne resultados; complemente com sua análise e capturas reais da aplicação."/><div className="checklist-grid">{['Definição do problema e tipo de aprendizado', 'Dataset e fluxo do projeto', 'Implementação e configuração', 'Métricas, matriz e validação cruzada', 'Análise crítica e limitações', 'Conclusão e referência ao código'].map((item, i) => <div key={item}><span>{String(i + 1).padStart(2, '0')}</span>{item}</div>)}</div><p className="small-muted">Para capturas de tela, use a ferramenta de captura do sistema enquanto a aplicação estiver executando e registre a tela real junto ao relatório acadêmico.</p></section></>}</>
}
