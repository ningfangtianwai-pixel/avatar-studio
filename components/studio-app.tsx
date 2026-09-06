'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import Image from 'next/image';
import {
  AlertCircle,
  AudioLines,
  CheckCircle2,
  CircleUserRound,
  Clapperboard,
  Clock3,
  Download,
  FolderOpen,
  Gauge,
  LayoutDashboard,
  LoaderCircle,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Settings2,
  Sparkles,
  Trash2,
  Upload,
  XCircle,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';

const API = 'http://127.0.0.1:8765';
type View =
  | 'dashboard'
  | 'avatars'
  | 'voices'
  | 'create'
  | 'videos'
  | 'settings';
type AvatarItem = { id: string; name: string; created_at: string };
type VoiceItem = {
  id: string;
  name: string;
  ref_text: string;
  created_at: string;
  audio_ready: boolean;
};
type JobItem = {
  id: string;
  title: string;
  subtitles: number;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  stage: string;
  progress: number;
  duration_seconds: number | null;
  error: string | null;
  created_at: string;
  completed_at: string | null;
  avatar_name: string;
  voice_name: string;
};
type StudioState = {
  avatars: AvatarItem[];
  voices: VoiceItem[];
  jobs: JobItem[];
  engine: {
    available: boolean;
    name: string;
    memory_mb: number;
    utilization: number;
  };
  media_root: string;
};
type WebMCPContext = {
  registerTool: (
    tool: Record<string, unknown>,
    options?: { signal?: AbortSignal },
  ) => void | Promise<void>;
};

const navigation: { id: View; label: string; icon: typeof LayoutDashboard }[] =
  [
    { id: 'dashboard', label: '工作台', icon: LayoutDashboard },
    { id: 'avatars', label: '数字人', icon: CircleUserRound },
    { id: 'voices', label: '声音', icon: AudioLines },
    { id: 'create', label: '新建口播', icon: Sparkles },
    { id: 'videos', label: '成片管理', icon: Clapperboard },
  ];

async function request<T = { ok: boolean }>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set('X-Avatar-Studio', '1');
  const response = await fetch(`${API}${path}`, { ...init, headers });
  if (!response.ok) {
    const body = (await response
      .json()
      .catch(() => ({ detail: '请求失败' }))) as { detail?: unknown };
    const detail = typeof body.detail === 'string'
      ? body.detail
      : Array.isArray(body.detail)
        ? body.detail.map((item: { msg?: string }) => item.msg || '输入不正确').join('；')
        : '请求失败';
    throw new Error(detail);
  }
  return response.json();
}

function durationLabel(seconds: number | null) {
  if (!seconds) return '--:--';
  const rounded = Math.round(seconds);
  return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, '0')}`;
}

function statusMeta(status: JobItem['status']) {
  if (status === 'completed')
    return {
      label: '完成',
      className: 'bg-emerald-50 text-emerald-700',
      icon: CheckCircle2,
    };
  if (status === 'running')
    return {
      label: '生成中',
      className: 'bg-blue-50 text-blue-700',
      icon: LoaderCircle,
    };
  if (status === 'queued')
    return {
      label: '排队中',
      className: 'bg-amber-50 text-amber-700',
      icon: Clock3,
    };
  if (status === 'failed')
    return {
      label: '失败',
      className: 'bg-red-50 text-red-700',
      icon: XCircle,
    };
  return {
    label: '已取消',
    className: 'bg-zinc-100 text-zinc-600',
    icon: Pause,
  };
}

export function StudioApp() {
  const [view, setView] = useState<View>('dashboard');
  const [state, setState] = useState<StudioState | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [avatarDialog, setAvatarDialog] = useState(false);
  const [voiceDialog, setVoiceDialog] = useState(false);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setState(await request<StudioState>('/api/state'));
      setError('');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '无法连接本地服务');
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void refresh(), 0);
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refresh]);

  useEffect(() => {
    const context = (document as Document & { modelContext?: WebMCPContext })
      .modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    const tools = [
      {
        name: 'read_avatar_studio_state',
        title: '读取数字人工作台状态',
        description: '读取当前数字人、声音、视频任务和本地 GPU 状态。',
        inputSchema: {
          type: 'object',
          properties: {},
          additionalProperties: false,
        },
        annotations: { readOnlyHint: true, untrustedContentHint: false },
        async execute() {
          const current = await request<StudioState>('/api/state');
          setState(current);
          return {
            avatars: current.avatars.length,
            voices: current.voices.length,
            jobs: current.jobs.length,
            gpu: current.engine,
          };
        },
      },
      {
        name: 'start_talking_video_creation',
        title: '打开新建口播',
        description: '在可见界面中打开新建数字人口播视频表单。',
        inputSchema: {
          type: 'object',
          properties: {},
          additionalProperties: false,
        },
        annotations: { readOnlyHint: true, untrustedContentHint: false },
        async execute() {
          setView('create');
          return { view: 'create' };
        },
      },
    ];
    for (const tool of tools) {
      try {
        void Promise.resolve(
          context.registerTool(tool, { signal: lifecycle.signal }),
        ).catch(() => undefined);
      } catch {
        /* WebMCP is optional in ordinary browsers. */
      }
    }
    return () => lifecycle.abort();
  }, []);

  const activeJob = useMemo(
    () =>
      state?.jobs.find(
        (job) => job.status === 'running' || job.status === 'queued',
      ),
    [state],
  );
  const completedJobs = useMemo(
    () => state?.jobs.filter((job) => job.status === 'completed') ?? [],
    [state],
  );

  async function submitAsset(
    kind: 'avatars' | 'voices',
    form: HTMLFormElement,
  ) {
    setBusy(true);
    try {
      await request(`/api/${kind}`, {
        method: 'POST',
        body: new FormData(form),
      });
      setAvatarDialog(false);
      setVoiceDialog(false);
      setNotice(kind === 'avatars' ? '数字人已加入素材库' : '声音已加入素材库');
      await refresh();
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : '保存失败');
    } finally {
      setBusy(false);
    }
  }

  async function remove(
    kind: 'avatars' | 'voices' | 'jobs',
    id: string,
    label: string,
  ) {
    if (!window.confirm(`确认删除“${label}”？文件会移入工作台回收站。`)) return;
    try {
      await request(`/api/${kind}/${id}`, { method: 'DELETE' });
      setNotice('已移入回收站');
      await refresh();
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : '删除失败');
    }
  }

  async function normalizeVoice(id: string) {
    setNotice('正在转换为兼容 WAV…');
    try {
      await request(`/api/voices/${id}/normalize`, { method: 'POST' });
      setNotice('声音格式已适配，可以用于配音');
      await refresh();
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : '转码失败');
    }
  }

  return (
    <main className="min-h-screen bg-[#f6f6f3] text-[#191918]">
      <div className="mx-auto flex min-h-screen max-w-[1560px]">
        <aside className="sticky top-0 hidden h-screen w-[236px] shrink-0 flex-col border-r border-black/[0.07] px-5 py-6 lg:flex">
          <button
            className="flex items-center gap-3 px-2 text-left"
            onClick={() => setView('dashboard')}
            type="button"
          >
            <div className="grid size-9 place-items-center rounded-xl bg-[#191918] text-white">
              <Sparkles className="size-4" />
            </div>
            <div>
              <p className="text-sm font-semibold tracking-tight">Avatar Studio</p>
              <p className="text-xs text-[#777772]">数字人工作台</p>
            </div>
          </button>
          <nav className="mt-10 space-y-1">
            {navigation.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm transition-colors ${view === id ? 'bg-white font-medium shadow-[0_1px_0_rgba(0,0,0,0.05)]' : 'text-[#696964] hover:bg-white/70 hover:text-[#191918]'}`}
                onClick={() => setView(id)}
                type="button"
              >
                <Icon className="size-[17px]" strokeWidth={1.8} />
                {label}
              </button>
            ))}
          </nav>
          <EngineCard state={state} activeJob={activeJob} />
          <button
            className="mt-3 flex items-center gap-3 px-3 py-2 text-sm text-[#777772]"
            onClick={() => setView('settings')}
            type="button"
          >
            <Settings2 className="size-[17px]" />
            设置
          </button>
        </aside>

        <section className="min-w-0 flex-1">
          <header className="flex h-[74px] items-center justify-between border-b border-black/[0.07] px-5 sm:px-8 xl:px-11">
            <div>
              <h1 className="text-[15px] font-semibold tracking-tight sm:text-base">
                数字人口播工作台
              </h1>
              <p className="mt-0.5 hidden text-xs text-[#85857f] sm:block">
                形象、声音与口播成片，都在本地完成
              </p>
            </div>
            <Button
              className="h-9 rounded-xl bg-[#191918] px-4 text-xs shadow-none hover:bg-black"
              onClick={() => setView('create')}
            >
              <Plus className="mr-1.5 size-4" />
              新建口播视频
            </Button>
          </header>
          <div className="border-b border-black/[0.07] px-5 py-2 lg:hidden">
            <div className="flex gap-1 overflow-x-auto">
              {navigation.map(({ id, label }) => (
                <button
                  key={id}
                  onClick={() => setView(id)}
                  className={`shrink-0 rounded-lg px-3 py-2 text-xs ${view === id ? 'bg-white font-medium' : 'text-[#777772]'}`}
                  type="button"
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          {error && (
            <div className="mx-5 mt-5 flex items-center gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-xs text-red-700 sm:mx-8 xl:mx-11">
              <AlertCircle className="size-4" />
              后端服务未连接：{error}
            </div>
          )}
          {notice && (
            <button
              className="fixed bottom-5 right-5 z-50 rounded-xl bg-[#20201f] px-4 py-3 text-xs text-white shadow-xl"
              onClick={() => setNotice('')}
              type="button"
            >
              {notice}
            </button>
          )}
          {view === 'dashboard' && (
            <Dashboard
              state={state}
              activeJob={activeJob}
              completedJobs={completedJobs}
              setView={setView}
            />
          )}
          {view === 'avatars' && (
            <AvatarLibrary
              items={state?.avatars ?? []}
              onAdd={() => setAvatarDialog(true)}
              onRemove={(id, name) => void remove('avatars', id, name)}
            />
          )}
          {view === 'voices' && (
            <VoiceLibrary
              items={state?.voices ?? []}
              onAdd={() => setVoiceDialog(true)}
              onNormalize={(id) => void normalizeVoice(id)}
              onRemove={(id, name) => void remove('voices', id, name)}
            />
          )}
          {view === 'create' && (
            <CreateVideo
              state={state}
              onCreated={async () => {
                await refresh();
                setView('videos');
                setNotice('任务已加入本地生成队列');
              }}
            />
          )}
          {view === 'videos' && (
            <VideoLibrary
              jobs={state?.jobs ?? []}
              refresh={refresh}
              onRemove={(id, title) => void remove('jobs', id, title)}
            />
          )}
          {view === 'settings' && <SettingsView state={state} />}
        </section>
      </div>

      <AssetDialog
        kind="avatar"
        open={avatarDialog}
        setOpen={setAvatarDialog}
        busy={busy}
        onSubmit={(form) => submitAsset('avatars', form)}
      />
      <AssetDialog
        kind="voice"
        open={voiceDialog}
        setOpen={setVoiceDialog}
        busy={busy}
        onSubmit={(form) => submitAsset('voices', form)}
      />
    </main>
  );
}

function AssetDialog({
  kind,
  open,
  setOpen,
  busy,
  onSubmit,
}: {
  kind: 'avatar' | 'voice';
  open: boolean;
  setOpen: (value: boolean) => void;
  busy: boolean;
  onSubmit: (form: HTMLFormElement) => Promise<void>;
}) {
  const isAvatar = kind === 'avatar';
  const prefix = isAvatar ? 'avatar' : 'voice';
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="rounded-2xl p-5 sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{isAvatar ? '新建数字人' : '新建声音'}</DialogTitle>
          <DialogDescription>
            {isAvatar
              ? '上传正面、清晰、机位稳定的视频。建议 10–30 秒，最长 120 秒、最大 256 MB，自动适配恒定 25 帧。'
              : '上传 8–20 秒干净人声，最长 60 秒、最大 256 MB，并填写逐字对应的原文。'}
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void onSubmit(event.currentTarget);
          }}
        >
          <label
            htmlFor={`${prefix}-name`}
            className="mt-2 block text-xs font-medium"
          >
            名称
          </label>
          <Input
            id={`${prefix}-name`}
            name="name"
            maxLength={120}
            required
            className="mt-2 h-10 rounded-xl"
            placeholder={
              isAvatar ? '例如：我的形象 · 正面半身' : '例如：我的声音 · 自然口播'
            }
          />
          <label
            htmlFor={`${prefix}-file`}
            className="mt-4 block text-xs font-medium"
          >
            {isAvatar ? '参考视频' : '参考音频'}
          </label>
          <Input
            id={`${prefix}-file`}
            name="file"
            type="file"
            required
            accept={
              isAvatar
                ? 'video/mp4,video/quicktime,video/webm,.mkv'
                : 'audio/*,.wav,.mp3,.m4a,.flac,.ogg'
            }
            className="mt-2 h-11 rounded-xl pt-2"
          />
          {!isAvatar && (
            <>
              <label
                htmlFor="voice-ref-text"
                className="mt-4 block text-xs font-medium"
              >
                参考音频原文
              </label>
              <Textarea
                id="voice-ref-text"
                name="ref_text"
                maxLength={2000}
                required
                rows={4}
                className="mt-2 rounded-xl"
                placeholder="把参考音频说的内容逐字写在这里"
              />
            </>
          )}
          <DialogFooter className="mt-6">
            <Button type="submit" disabled={busy} className="rounded-xl">
              {busy && <LoaderCircle className="mr-2 size-4 animate-spin" />}
              保存{isAvatar ? '数字人' : '声音'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function PageIntro({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
      <div>
        <p className="text-xs font-medium uppercase tracking-[0.14em] text-[#888882]">
          {eyebrow}
        </p>
        <h2 className="mt-2 text-2xl font-semibold tracking-[-0.035em] sm:text-[30px]">
          {title}
        </h2>
        <p className="mt-2 text-sm text-[#777772]">{description}</p>
      </div>
      {action}
    </div>
  );
}

function EngineCard({
  state,
  activeJob,
}: {
  state: StudioState | null;
  activeJob?: JobItem;
}) {
  const engine = state?.engine;
  return (
    <div className="mt-auto rounded-2xl border border-black/[0.07] bg-white/55 p-4">
      <div className="flex items-center justify-between text-xs">
        <span className="text-[#777772]">本地 GPU</span>
        <span className="inline-flex items-center gap-1.5 font-medium">
          <i
            className={`size-1.5 rounded-full ${engine?.available ? (activeJob ? 'bg-blue-500' : 'bg-emerald-500') : 'bg-red-500'}`}
          />
          {activeJob ? '运行中' : engine?.available ? '空闲' : '离线'}
        </span>
      </div>
      <p className="mt-3 truncate text-sm font-medium">
        {engine?.name ?? '正在检测'}
      </p>
      <p className="mt-1 text-xs leading-5 text-[#888882]">
        {engine?.memory_mb
          ? `${Math.round(engine.memory_mb / 1024)}GB 显存 · ${engine.utilization}%`
          : '任务只在这台电脑上运行'}
      </p>
    </div>
  );
}

function Dashboard({
  state,
  activeJob,
  completedJobs,
  setView,
}: {
  state: StudioState | null;
  activeJob?: JobItem;
  completedJobs: JobItem[];
  setView: (view: View) => void;
}) {
  const metrics = [
    { label: '数字人形象', value: state?.avatars.length ?? 0, note: '已就绪' },
    { label: '可用声音', value: state?.voices.length ?? 0, note: 'Qwen3-TTS' },
    { label: '已生成成片', value: completedJobs.length, note: '本地保存' },
  ];
  return (
    <div className="px-5 py-7 sm:px-8 xl:px-11 xl:py-10">
      <PageIntro
        eyebrow="Workspace"
        title="开始今天的内容生产"
        description="选择数字人和声音，粘贴文案，然后交给本地显卡。"
        action={
          <Button
            variant="outline"
            className="w-fit rounded-xl border-black/10 bg-white text-xs shadow-none"
            onClick={() =>
              void request('/api/open-media-folder', { method: 'POST' })
            }
          >
            <FolderOpen className="mr-1.5 size-4" />
            打开媒体目录
          </Button>
        }
      />
      <div className="mt-8 grid gap-3 sm:grid-cols-3">
        {metrics.map((item) => (
          <article
            key={item.label}
            className="rounded-2xl border border-black/[0.07] bg-white p-5"
          >
            <p className="text-xs text-[#7c7c76]">{item.label}</p>
            <div className="mt-4 flex items-end justify-between">
              <strong className="text-3xl font-semibold tracking-[-0.04em]">
                {item.value}
              </strong>
              <span className="pb-1 text-xs text-[#8b8b85]">{item.note}</span>
            </div>
          </article>
        ))}
      </div>
      <div className="mt-7 grid gap-5 xl:grid-cols-[1.25fr_0.75fr]">
        <section className="overflow-hidden rounded-2xl border border-black/[0.07] bg-white">
          <div className="flex items-center justify-between border-b border-black/[0.06] px-5 py-4">
            <div>
              <h3 className="text-sm font-semibold">最近成片</h3>
              <p className="mt-1 text-xs text-[#85857f]">
                生成完成的视频会自动出现在这里
              </p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="text-xs"
              onClick={() => setView('videos')}
            >
              查看全部
            </Button>
          </div>
          <div className="space-y-3 p-4 sm:p-5">
            {completedJobs.slice(0, 2).map((job) => (
              <VideoRow key={job.id} job={job} />
            ))}
            {completedJobs.length === 0 && (
              <Empty
                label="还没有成片"
                action="新建第一个口播"
                onClick={() => setView('create')}
              />
            )}
          </div>
        </section>
        <section className="rounded-2xl bg-[#222220] p-5 text-white sm:p-6">
          <div className="flex items-center justify-between">
            <div className="grid size-9 place-items-center rounded-xl bg-white/10">
              <Gauge className="size-[18px]" />
            </div>
            <Badge className="bg-white/10 text-[10px] text-white">
              单任务队列
            </Badge>
          </div>
          <h3 className="mt-7 text-base font-semibold">
            {activeJob ? activeJob.stage : '生产引擎已就绪'}
          </h3>
          <p className="mt-2 text-xs leading-5 text-white/55">
            {activeJob
              ? activeJob.title
              : '连续长句配音、平滑口型驱动、恒定帧率与双行字幕一次完成。'}
          </p>
          <div className="mt-7">
            <div className="mb-2 flex justify-between text-[11px] text-white/55">
              <span>{activeJob ? '当前进度' : 'GPU 占用'}</span>
              <span>
                {activeJob?.progress ?? state?.engine.utilization ?? 0}%
              </span>
            </div>
            <Progress
              value={activeJob?.progress ?? state?.engine.utilization ?? 0}
              className="h-1.5 bg-white/10"
            />
          </div>
          <Button
            className="mt-7 h-10 w-full rounded-xl bg-white text-xs text-[#191918] hover:bg-white/90"
            onClick={() => setView(activeJob ? 'videos' : 'create')}
          >
            <Sparkles className="mr-1.5 size-4" />
            {activeJob ? '查看生成任务' : '开始新口播'}
          </Button>
        </section>
      </div>
    </div>
  );
}

function AvatarLibrary({
  items,
  onAdd,
  onRemove,
}: {
  items: AvatarItem[];
  onAdd: () => void;
  onRemove: (id: string, name: string) => void;
}) {
  return (
    <div className="px-5 py-7 sm:px-8 xl:px-11 xl:py-10">
      <PageIntro
        eyebrow="Avatars"
        title="数字人形象"
        description="管理可复用的出镜形象。删除后文件会进入工作台回收站。"
        action={
          <Button onClick={onAdd} className="rounded-xl">
            <Upload className="mr-2 size-4" />
            新建数字人
          </Button>
        }
      />
      <div className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {items.map((item) => (
          <article
            key={item.id}
            className="overflow-hidden rounded-2xl border border-black/[0.07] bg-white"
          >
            <div className="relative aspect-video bg-[#deddd7]">
              <Image
                unoptimized
                width={640}
                height={360}
                src={`${API}/api/media/avatar/${item.id}`}
                alt={item.name}
                className="h-full w-full object-cover"
              />
            </div>
            <div className="flex items-center justify-between p-4">
              <div>
                <h3 className="text-sm font-medium">{item.name}</h3>
                <p className="mt-1 text-xs text-[#888882]">已可用于口播生成</p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="size-8 text-[#888882] hover:text-red-600"
                onClick={() => onRemove(item.id, item.name)}
              >
                <Trash2 className="size-4" />
              </Button>
            </div>
          </article>
        ))}
        <button
          onClick={onAdd}
          className="grid min-h-56 place-items-center rounded-2xl border border-dashed border-black/15 bg-white/35 text-sm text-[#777772] hover:bg-white"
          type="button"
        >
          <span className="flex flex-col items-center gap-3">
            <span className="grid size-10 place-items-center rounded-full bg-black/[0.05]">
              <Plus className="size-4" />
            </span>
            添加新形象
          </span>
        </button>
      </div>
    </div>
  );
}

function VoiceLibrary({
  items,
  onAdd,
  onNormalize,
  onRemove,
}: {
  items: VoiceItem[];
  onAdd: () => void;
  onNormalize: (id: string) => void;
  onRemove: (id: string, name: string) => void;
}) {
  const [playing, setPlaying] = useState<string | null>(null);
  function play(id: string) {
    const audio = new Audio(`${API}/api/media/voice/${id}`);
    setPlaying(id);
    void audio.play();
    audio.onended = () => setPlaying(null);
  }
  return (
    <div className="px-5 py-7 sm:px-8 xl:px-11 xl:py-10">
      <PageIntro
        eyebrow="Voices"
        title="声音库"
        description="上传时会自动转成兼容 WAV，也可以随时一键重新适配。"
        action={
          <Button onClick={onAdd} className="rounded-xl">
            <Upload className="mr-2 size-4" />
            新建声音
          </Button>
        }
      />
      <div className="mt-8 grid gap-4 lg:grid-cols-2">
        {items.map((item) => (
          <article
            key={item.id}
            className="rounded-2xl border border-black/[0.07] bg-white p-5"
          >
            <div className="flex items-start gap-4">
              <button
                onClick={() => play(item.id)}
                className="grid size-11 shrink-0 place-items-center rounded-full bg-[#20201f] text-white"
                type="button"
              >
                {playing === item.id ? (
                  <Pause className="size-4" />
                ) : (
                  <Play className="ml-0.5 size-4 fill-current" />
                )}
              </button>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-medium">{item.name}</h3>
                  <Badge
                    variant="secondary"
                    className={`rounded-md px-1.5 text-[10px] ${item.audio_ready ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'}`}
                  >
                    {item.audio_ready ? 'WAV 已适配' : '需要转码'}
                  </Badge>
                </div>
                <p className="mt-2 line-clamp-2 text-xs leading-5 text-[#888882]">
                  {item.ref_text}
                </p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="size-8 text-[#888882] hover:text-red-600"
                onClick={() => onRemove(item.id, item.name)}
              >
                <Trash2 className="size-4" />
              </Button>
            </div>
            <div className="mt-5 h-9 rounded-lg bg-[repeating-linear-gradient(90deg,#deded8_0_2px,transparent_2px_6px)] opacity-70" />
            <Button
              variant="outline"
              size="sm"
              className="mt-4 rounded-lg text-xs"
              onClick={() => onNormalize(item.id)}
            >
              <RefreshCw className="mr-1.5 size-3.5" />
              {item.audio_ready ? '重新适配' : '一键转码'}
            </Button>
          </article>
        ))}
      </div>
    </div>
  );
}

function CreateVideo({
  state,
  onCreated,
}: {
  state: StudioState | null;
  onCreated: () => Promise<void>;
}) {
  const [subtitles, setSubtitles] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState('');
  async function submit(form: HTMLFormElement) {
    setSubmitting(true);
    setMessage('');
    const data = new FormData(form);
    try {
      await request('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: data.get('title'),
          script_text: data.get('script_text'),
          avatar_id: data.get('avatar_id'),
          voice_id: data.get('voice_id'),
          subtitles,
        }),
      });
      await onCreated();
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : '创建失败');
    } finally {
      setSubmitting(false);
    }
  }
  return (
    <div className="px-5 py-7 sm:px-8 xl:px-11 xl:py-10">
      <PageIntro
        eyebrow="New video"
        title="新建口播视频"
        description="创建后会进入单任务队列。生成耗时取决于显卡、文案长度与参考素材。"
      />
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void submit(event.currentTarget);
        }}
        className="mt-8 grid gap-5 xl:grid-cols-[1fr_340px]"
      >
        <section className="rounded-2xl border border-black/[0.07] bg-white p-5 sm:p-6">
          <label htmlFor="video-title" className="text-xs font-medium">
            视频标题
          </label>
          <Input
            id="video-title"
            name="title"
            maxLength={120}
            required
            className="mt-2 h-11 rounded-xl"
            placeholder="例如：产品介绍 · 精简版"
          />
          <label
            htmlFor="video-script"
            className="mt-6 block text-xs font-medium"
          >
            口播文案
          </label>
          <Textarea
            id="video-script"
            name="script_text"
            maxLength={10000}
            required
            className="mt-2 min-h-[360px] resize-y rounded-xl p-4 leading-7"
            placeholder="把口播文案粘贴到这里。系统会合并过短段落、生成连续配音、平滑驱动口型并智能排版字幕。"
          />
        </section>
        <aside className="h-fit rounded-2xl border border-black/[0.07] bg-white p-5">
          <h3 className="text-sm font-semibold">生成配置</h3>
          <label
            htmlFor="video-avatar"
            className="mt-5 block text-xs font-medium"
          >
            数字人
          </label>
          <select
            id="video-avatar"
            name="avatar_id"
            required
            defaultValue={state?.avatars[0]?.id}
            className="mt-2 h-11 w-full rounded-xl border border-black/10 bg-white px-3 text-sm"
          >
            {state?.avatars.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
          <label
            htmlFor="video-voice"
            className="mt-5 block text-xs font-medium"
          >
            声音
          </label>
          <select
            id="video-voice"
            name="voice_id"
            required
            defaultValue={state?.voices[0]?.id}
            className="mt-2 h-11 w-full rounded-xl border border-black/10 bg-white px-3 text-sm"
          >
            {state?.voices.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
          <div className="mt-6 flex items-center justify-between rounded-xl bg-[#f5f5f2] p-4">
            <div>
              <p className="text-xs font-medium">自动字幕</p>
              <p className="mt-1 text-[11px] text-[#888882]">
                双行大字幕，自动保护数字与英文
              </p>
            </div>
            <Switch checked={subtitles} onCheckedChange={setSubtitles} />
          </div>
          <div className="mt-3 flex items-start gap-3 rounded-xl border border-emerald-100 bg-emerald-50/70 p-4">
            <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-700" />
            <div>
              <p className="text-xs font-medium text-emerald-900">
                质量优化已默认开启
              </p>
              <p className="mt-1 text-[11px] leading-5 text-emerald-800/70">
                连贯配音 · 恒定 25fps · 音量均衡 · 单次高清合成
              </p>
            </div>
          </div>
          {message && <p className="mt-4 text-xs text-red-600">{message}</p>}
          <Button
            type="submit"
            disabled={
              submitting || !state?.avatars.length || !state?.voices.length
            }
            className="mt-6 h-11 w-full rounded-xl"
          >
            <Sparkles className="mr-2 size-4" />
            {submitting ? '正在创建…' : '加入生成队列'}
          </Button>
          <p className="mt-3 text-center text-[11px] text-[#999993]">
            本地运行，不上传云端，不产生模型 API 费用
          </p>
        </aside>
      </form>
    </div>
  );
}

function VideoLibrary({
  jobs,
  refresh,
  onRemove,
}: {
  jobs: JobItem[];
  refresh: () => Promise<void>;
  onRemove: (id: string, title: string) => void;
}) {
  const [actionError, setActionError] = useState('');
  const [pendingJob, setPendingJob] = useState<string | null>(null);
  async function jobAction(id: string, action: 'cancel' | 'retry') {
    setPendingJob(id);
    setActionError('');
    try {
      await request(`/api/jobs/${id}/${action}`, { method: 'POST' });
      await refresh();
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : '操作失败');
    } finally {
      setPendingJob(null);
    }
  }
  return (
    <div className="px-5 py-7 sm:px-8 xl:px-11 xl:py-10">
      <PageIntro
        eyebrow="Videos"
        title="成片与任务"
        description="查看排队、生成中、已完成和失败的口播任务。"
      />
      {actionError && <p role="alert" className="mt-4 text-sm text-red-600">{actionError}</p>}
      <div className="mt-8 overflow-hidden rounded-2xl border border-black/[0.07] bg-white">
        <div className="divide-y divide-black/[0.06]">
          {jobs.map((job) => (
            <div key={job.id} className="p-4 sm:p-5">
              <div className="grid gap-4 sm:grid-cols-[150px_1fr_auto] sm:items-center">
                {job.status === 'completed' ? (
                  <video
                    className="aspect-video w-full rounded-xl bg-black object-cover"
                    src={`${API}/api/media/video/${job.id}`}
                    preload="metadata"
                    controls
                  >
                    <track kind="captions" />
                  </video>
                ) : (
                  <div className="grid aspect-video place-items-center rounded-xl bg-[#efefeb]">
                    <Clapperboard className="size-5 text-[#aaa9a2]" />
                  </div>
                )}
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="truncate text-sm font-medium">
                      {job.title}
                    </h3>
                    <StatusBadge status={job.status} />
                  </div>
                  <p className="mt-2 text-xs text-[#85857f]">
                    {job.avatar_name} · {job.voice_name} ·{' '}
                    {job.subtitles ? '带字幕' : '无字幕'}
                  </p>
                  <p className="mt-1 text-xs text-[#aaa9a2]">
                    {job.stage}
                    {job.duration_seconds
                      ? ` · ${durationLabel(job.duration_seconds)}`
                      : ''}
                  </p>
                  {(job.status === 'running' || job.status === 'queued') && (
                    <Progress
                      value={job.progress}
                      className="mt-3 h-1.5 max-w-md"
                    />
                  )}
                  {job.error && (
                    <p className="mt-2 text-xs text-red-600">{job.error}</p>
                  )}
                </div>
                <div className="flex gap-1">
                  {job.status === 'completed' && (
                    <a href={`${API}/api/media/video/${job.id}`} download>
                      <Button variant="ghost" size="icon" className="size-9">
                        <Download className="size-4" />
                      </Button>
                    </a>
                  )}
                  {job.status === 'queued' && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-xs"
                      disabled={pendingJob !== null}
                      onClick={() => void jobAction(job.id, 'cancel')}
                    >
                      取消
                    </Button>
                  )}
                  {(job.status === 'failed' || job.status === 'cancelled') && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs"
                      disabled={pendingJob !== null}
                      onClick={() => void jobAction(job.id, 'retry')}
                    >
                      重新生成
                    </Button>
                  )}
                  {!['running', 'queued'].includes(job.status) && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-9 text-[#888882] hover:text-red-600"
                      onClick={() => onRemove(job.id, job.title)}
                    >
                      <Trash2 className="size-4" />
                    </Button>
                  )}
                </div>
              </div>
            </div>
          ))}
          {jobs.length === 0 && (
            <div className="p-6">
              <Empty label="还没有任务" action="去创建口播" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function VideoRow({ job }: { job: JobItem }) {
  return (
    <article className="grid gap-4 rounded-xl border border-black/[0.06] p-3 sm:grid-cols-[148px_1fr] sm:items-center">
      <video
        className="aspect-video w-full rounded-lg bg-black object-cover"
        src={`${API}/api/media/video/${job.id}`}
        preload="metadata"
        controls
      >
        <track kind="captions" />
      </video>
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <h4 className="truncate text-sm font-medium">{job.title}</h4>
          <StatusBadge status={job.status} />
        </div>
        <p className="mt-2 text-xs text-[#85857f]">
          {durationLabel(job.duration_seconds)} ·{' '}
          {job.subtitles ? '带字幕' : '无字幕'}
        </p>
        <p className="mt-1 text-xs text-[#aaa9a2]">
          {job.avatar_name} · {job.voice_name}
        </p>
      </div>
    </article>
  );
}
function StatusBadge({ status }: { status: JobItem['status'] }) {
  const meta = statusMeta(status);
  const Icon = meta.icon;
  return (
    <Badge
      variant="secondary"
      className={`rounded-md px-1.5 text-[10px] font-medium ${meta.className}`}
    >
      <Icon
        className={`mr-1 size-3 ${status === 'running' ? 'animate-spin' : ''}`}
      />
      {meta.label}
    </Badge>
  );
}
function SettingsView({ state }: { state: StudioState | null }) {
  return (
    <div className="px-5 py-7 sm:px-8 xl:px-11 xl:py-10">
      <PageIntro
        eyebrow="Settings"
        title="本地设置"
        description="本地目录与单 GPU 队列。Manny 与 Codex 协作完成 · MIT。"
      />
      <div className="mt-8 max-w-3xl space-y-4">
        <section className="rounded-2xl border border-black/[0.07] bg-white p-5">
          <h3 className="text-sm font-medium">媒体目录</h3>
          <code className="mt-3 block rounded-xl bg-[#f4f4f1] p-4 text-xs text-[#666660]">
            {state?.media_root ?? '正在读取…'}
          </code>
        </section>
        <section className="rounded-2xl border border-black/[0.07] bg-white p-5">
          <h3 className="text-sm font-medium">生成引擎</h3>
          <div className="mt-3 grid gap-3 text-xs sm:grid-cols-3">
            <p>
              <span className="text-[#888882]">显卡</span>
              <br />
              <b className="mt-1 inline-block font-medium">
                {state?.engine.name ?? '检测中'}
              </b>
            </p>
            <p>
              <span className="text-[#888882]">配音</span>
              <br />
              <b className="mt-1 inline-block font-medium">Qwen3-TTS 0.6B</b>
            </p>
            <p>
              <span className="text-[#888882]">口型</span>
              <br />
              <b className="mt-1 inline-block font-medium">MuseTalk 1.5</b>
            </p>
          </div>
        </section>
        <section className="rounded-2xl border border-amber-200 bg-amber-50 p-5 text-xs leading-6 text-amber-900">
          为了避免显存冲突，工作台一次只生成一个视频。关闭网页不等于停止后端；前台启动时请保持服务终端运行。后台常驻需单独配置。
        </section>
        <section className="rounded-2xl border border-black/[0.07] bg-white p-5 text-sm leading-7">
          <h3 className="font-medium">关于数字人口播工作台</h3>
          <p className="mt-3 text-[#666660]">
            本系统由 Manny 与 Codex 协作完成，将形象管理、声音克隆、口播生成与成片管理整合为本地工作流。
            已加入语义段配音、停顿与音量平滑、章节动作衔接及统一帧率合成。
          </p>
          <p className="mt-2 text-[#666660]">
            作者与维护者：Manny。原创工作台代码采用 MIT；Qwen3-TTS、MuseTalk 等遵循各自许可。本项目不代表 OpenAI 官方产品或背书。
          </p>
          <p className="mt-2 text-[#666660]">
            字幕仍为近似时间对齐；情绪控制尚未实现，模型抖动和循环动作仍可能出现。
          </p>
          <p className="mt-3">
            如果有任何问题，欢迎大家反馈：{' '}
            <a className="break-all underline underline-offset-4" href="mailto:ningfangtianwai@gmail.com">ningfangtianwai@gmail.com</a>
          </p>
        </section>
      </div>
    </div>
  );
}
function Empty({
  label,
  action,
  onClick,
}: {
  label: string;
  action: string;
  onClick?: () => void;
}) {
  return (
    <div className="grid min-h-40 place-items-center text-center">
      <div>
        <div className="mx-auto grid size-10 place-items-center rounded-full bg-black/[0.05]">
          <Clapperboard className="size-4 text-[#888882]" />
        </div>
        <p className="mt-3 text-sm font-medium">{label}</p>
        <button
          onClick={onClick}
          className="mt-2 text-xs text-[#777772] underline underline-offset-4"
          type="button"
        >
          {action}
        </button>
      </div>
    </div>
  );
}
