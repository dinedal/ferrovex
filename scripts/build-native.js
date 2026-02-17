const { spawnSync } = require('node:child_process')

const disableMetal = process.env.FERROVEX_DISABLE_METAL === '1'
const enableMetal = process.platform === 'darwin' && !disableMetal
const features = enableMetal ? 'onnx,metal' : 'onnx'
const isRelease = process.argv.includes('--release')
const passthroughArgs = process.argv.slice(2).filter((arg) => arg !== '--release')
const args = ['build', '--platform', '--features', features, ...passthroughArgs]
const napiCliEntry = require.resolve('@napi-rs/cli/scripts/index.js')

if (isRelease) {
  args.push('--release')
}

const result = spawnSync(process.execPath, [napiCliEntry, ...args], { stdio: 'inherit' })

if (result.error) {
  console.error(result.error.message)
  process.exit(1)
}

process.exit(result.status ?? 1)
