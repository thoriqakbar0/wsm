// Twist — a Texture by Textures · https://textures.dev/s/twist
// Install another: npx shadcn@latest add https://textures.dev/r/<slug>.json
// For agents: https://textures.md
// License: free to use
"use client"
import { type CSSProperties, useEffect, useRef } from "react"
function selfSize(root: HTMLElement | null): void {
	const host = root?.parentElement
	if (!root || !host || host.getBoundingClientRect().height > 0) return
	root.style.minHeight = "240px"
}
type Pointer = { x: number; y: number; active: number; reduced: boolean }
const AMBIENT_SPEED = 1.5
const AMBIENT_CYCLE = 4.5
type EngineOptions = { ambientSpeed?: number }
function engine(
	root: HTMLElement,
	draw: (p: Pointer, dt: number) => boolean | void,
	min: [number, number],
	options: EngineOptions = {},
): () => void {
	const svg = root.querySelector("svg")
	if (!svg) return () => {}
	const mq = () => matchMedia("(prefers-reduced-motion: reduce)").matches || matchMedia("(update: slow)").matches
	const touchMq = matchMedia("(hover: none), (pointer: coarse)")
	const forced = () => root.closest("[data-motion]")?.getAttribute("data-motion") ?? null
	const wantsAmbient = () => (forced() ? forced() === "ambient" : touchMq.matches)
	const speed = AMBIENT_SPEED * (options.ambientSpeed ?? 1)
	const w = (2 * Math.PI) / AMBIENT_CYCLE
	const phase = Math.random() * 2 * Math.PI
	let onScreen = false
	let ambient = false
	let clock = 0
	let hold = 0
	let box: { left: number; top: number; s: number; ox: number; oy: number } | null = null
	let px = 366
	let py = 258
	let cx = 0
	let cy = 0
	let active = 0
	let target = 0
	let vel = 0
	let frame = 0
	let last = 0
	let lastMoveAt = 0
	const measure = () => {
		const r = svg.getBoundingClientRect()
		if (r.width < min[0] || r.height < min[1]) {
			box = null
			return
		}
		const s = Math.max(r.width / 732, r.height / 516)
		box = { left: r.left, top: r.top, s, ox: (r.width - 732 * s) / 2, oy: (r.height - 516 * s) / 2 }
	}
	const tick = (now: number) => {
		const dt = Math.min(last ? (now - last) / 1000 : 1 / 60, 1 / 30)
		last = now
		let moving = false
		const reduced = mq()
		if (ambient && now > hold) {
			if (reduced) target = 0
			else {
				clock += dt * speed
				const t = clock
				px = 366 + 366 * (0.76 * Math.sin(w * t + phase) + 0.2 * Math.sin(1.93 * w * t + 1.2 + phase))
				py = 258 + 258 * (0.72 * Math.sin(0.75 * w * t + 0.8 + phase) + 0.2 * Math.sin(1.37 * w * t + 2.1 + phase))
				target = 1
				moving = true
			}
		}
		if (reduced) {
			active = target
			vel = 0
		} else {
			const decay = Math.exp(-30 * dt)
			const off = active - target
			const lead = vel + 30 * off
			active = target + (off + lead * dt) * decay
			vel = (vel - 30 * lead * dt) * decay
			if (Math.abs(vel) < 0.02 && Math.abs(active - target) < 0.02) {
				active = target
				vel = 0
			} else moving = true
		}
		const more = draw({ x: px, y: py, active, reduced }, dt)
		if (moving || more) frame = requestAnimationFrame(tick)
		else {
			frame = 0
			last = 0
		}
	}
	const wake = () => {
		if (!frame) frame = requestAnimationFrame(tick)
	}
	const touchIsAmbient = (e: Event) => (e as PointerEvent).pointerType === "touch" && touchMq.matches
	const onEnter = (e: PointerEvent) => {
		if (touchIsAmbient(e)) return
		measure()
	}
	const onMove = (e: PointerEvent) => {
		if (touchIsAmbient(e)) return
		const t = performance.now()
		if (!box || t - lastMoveAt > 250) measure()
		lastMoveAt = t
		if (!box) return
		if (!e.isTrusted) hold = performance.now() + 1000
		cx = e.clientX
		cy = e.clientY
		px = (cx - box.left - box.ox) / box.s
		py = (cy - box.top - box.oy) / box.s
		target = 1
		wake()
	}
	const onLeave = (e?: Event) => {
		if (e && touchIsAmbient(e)) return
		if (e && !e.isTrusted) hold = 0
		if (ambient && performance.now() > hold) return
		target = 0
		wake()
	}
	const onFocus = () => {
		measure()
		if (!box) return
		px = 366
		py = 258
		target = 1
		wake()
	}
	const onResize = () => {
		if (box || ambient) measure()
		if (ambient && !box) ambient = false
		settle()
	}
	const onScroll = () => {
		if (!box || target !== 1 || (ambient && performance.now() > hold)) return
		measure()
		if (!box) return
		px = (cx - box.left - box.ox) / box.s
		py = (cy - box.top - box.oy) / box.s
		wake()
	}
	const settle = () => {
		const next = onScreen && !document.hidden && wantsAmbient()
		if (next === ambient) return
		ambient = next
		if (ambient) {
			measure()
			if (!box) {
				ambient = false
				return
			}
			wake()
		} else {
			target = 0
			wake()
		}
	}
	const io =
		typeof IntersectionObserver === "function"
			? new IntersectionObserver(
					([entry]) => {
						onScreen = !!entry?.isIntersecting
						settle()
					},
					{ threshold: 0.2 },
				)
			: null
	io?.observe(svg)
	if (!io) {
		onScreen = true
		settle()
	}
	const onVisibility = () => settle()
	const onCapability = () => settle()
	const reducedMq = matchMedia("(prefers-reduced-motion: reduce)")
	const onReduced = () => wake()
	document.addEventListener("visibilitychange", onVisibility)
	touchMq.addEventListener?.("change", onCapability)
	reducedMq.addEventListener?.("change", onReduced)
	const host: Element = root.closest("a, button, [tabindex]") ?? root.parentElement ?? root
	root.addEventListener("pointerenter", onEnter)
	root.addEventListener("pointermove", onMove, { passive: true })
	root.addEventListener("pointerleave", onLeave)
	root.addEventListener("pointercancel", onLeave)
	host.addEventListener("focusin", onFocus)
	host.addEventListener("focusout", onLeave)
	window.addEventListener("resize", onResize, { passive: true })
	window.addEventListener("scroll", onScroll, { passive: true, capture: true })
	return () => {
		root.removeEventListener("pointerenter", onEnter)
		root.removeEventListener("pointermove", onMove)
		root.removeEventListener("pointerleave", onLeave)
		root.removeEventListener("pointercancel", onLeave)
		host.removeEventListener("focusin", onFocus)
		host.removeEventListener("focusout", onLeave)
		window.removeEventListener("resize", onResize)
		window.removeEventListener("scroll", onScroll, { capture: true })
		document.removeEventListener("visibilitychange", onVisibility)
		touchMq.removeEventListener?.("change", onCapability)
		reducedMq.removeEventListener?.("change", onReduced)
		io?.disconnect()
		if (frame) cancelAnimationFrame(frame)
	}
}
const W = 732
const H = 516
const COLS = 25
const ROWS = 18
const CELL_W = 27
const CELL_H = 26
const X0 = 17
const Y0 = 17
const PITCH_X = (714 - 17 - CELL_W) / (COLS - 1)
const PITCH_Y = (498 - 17 - CELL_H) / (ROWS - 1)
const T_MIN = 3.75
const T_MAX = 26.5
const BAR_LEN = 44
const CELLS: [number, number, number][] = []
for (let r = 0; r < ROWS; r++)
	for (let c = 0; c < COLS; c++)
		CELLS.push([Math.round((X0 + c * PITCH_X) * 100) / 100, Math.round((Y0 + r * PITCH_Y) * 100) / 100, CELL_W])
export type TwistProps = {
	tokens?: Partial<Record<"--texture-ink" | "--texture-field", string>>
	className?: string
	static?: boolean
}
const ALIGN = 3
const REACH = 260
const TAU_NEAR = 0.035
const TAU_FAR = 0.06
const TAU_HOME = 0.16
const DEAD_A = 0.6
const DEAD_S = 0.04
const WRITE_A = 0.2
const WRITE_S = 0.01
const arc = (deg: number) => (((deg % 180) + 270) % 180) - 90
export function Twist({ tokens, className = "", static: isStatic = false }: TwistProps) {
	const ref = useRef<HTMLDivElement>(null)
	useEffect(() => selfSize(ref.current), [])
	const bars = useRef<(SVGRectElement | null)[]>([])
	useEffect(() => {
		const root = ref.current
		if (!root || isStatic) return
		const n = CELLS.length
		const cx = new Float32Array(n)
		const cy = new Float32Array(n)
		for (let i = 0; i < n; i++) {
			const [x, y, w] = CELLS[i]!
			cx[i] = x + w / 2
			cy[i] = y + CELL_H / 2
		}
		const angle = new Float32Array(n)
		const thick = new Float32Array(n).fill(1)
		const wa = new Float32Array(n)
		const ws = new Float32Array(n).fill(1)
		const smax = T_MAX / T_MIN
		return engine(
			root,
			({ x, y, active, reduced }, dt) => {
				let moving = false
				for (let i = 0; i < n; i++) {
					const dx = x - cx[i]!
					const dy = y - cy[i]!
					const d = Math.sqrt(dx * dx + dy * dy)
					const bearing = (Math.atan2(dy, dx) * 180) / Math.PI - 90
					const ta = active > 0 ? arc(bearing) * active : 0
					const f = d < 1 ? 1 : Math.abs(Math.cos(2 * Math.atan2(dy, dx))) ** ALIGN
					const ts = 1 + (smax - 1) * f * active
					const a = angle[i]!
					const s = thick[i]!
					if (a === ta && s === ts) continue
					let na: number
					let ns: number
					if (reduced) {
						na = ta
						ns = ts
					} else {
						const tau = active > 0 ? TAU_NEAR + (TAU_FAR - TAU_NEAR) * Math.min(1, d / REACH) : TAU_HOME
						const k = 1 - Math.exp(-dt / tau)
						na = a + arc(ta - a) * k
						ns = s + (ts - s) * k
						if (Math.abs(arc(ta - na)) < DEAD_A) na = ta
						if (Math.abs(ts - ns) < DEAD_S) ns = ts
					}
					angle[i] = na
					thick[i] = ns
					if (na !== ta || ns !== ts) moving = true
					if (
						Math.abs(na - wa[i]!) > WRITE_A ||
						Math.abs(ns - ws[i]!) > WRITE_S ||
						(na === 0 && ns === 1 && (wa[i] !== 0 || ws[i] !== 1))
					) {
						wa[i] = na
						ws[i] = ns
						const el = bars.current[i]
						if (el)
							el.style.transform = na === 0 && ns === 1 ? "" : `rotate(${na.toFixed(2)}deg) scaleX(${ns.toFixed(3)})`
					}
				}
				return moving
			},
			[160, 112],
		)
	}, [isStatic])
	const style = {
		"--texture-ink": "var(--foreground)",
		"--texture-field": "var(--background)",
		...tokens,
		color: "var(--texture-ink)",
		background: "var(--texture-field)",
	} as CSSProperties
	return (
		<div
			ref={ref}
			style={{
				position: "relative",
				isolation: "isolate",
				height: "100%",
				width: "100%",
				overflow: "clip",
				contain: "layout paint style",
				containerType: "size",
				...style,
			}}
			data-texture="twist"
			className={className}
		>
			<svg
				viewBox={`0 0 ${W} ${H}`}
				preserveAspectRatio="xMidYMid slice"
				aria-hidden="true"
				focusable="false"
				fill="currentColor"
				style={{ position: "absolute", inset: 0, display: "block", height: "100%", width: "100%" }}
			>
				{CELLS.map(([x, y, w], i) => (
					<svg key={i} x={x} y={y} width={w} height={CELL_H} viewBox={`0 0 ${w} ${CELL_H}`} overflow="hidden">
						<rect
							ref={(el) => {
								bars.current[i] = el
							}}
							x={(w - T_MIN) / 2}
							y={(CELL_H - BAR_LEN) / 2}
							width={T_MIN}
							height={BAR_LEN}
							style={{ transformBox: "fill-box", transformOrigin: "center" }}
						/>
					</svg>
				))}
			</svg>
		</div>
	)
}
