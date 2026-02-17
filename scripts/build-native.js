const { spawnSync } = require('node:child_process')

const disableMetal = process.env.FERROVEX_DISABLE_METAL === '1'
const enableMetal = process.platform === 'darwin' && !disableMetal
const features = enableMetal ? 'onnx,metal' : 'onnx'
const isRelease = process.argv.includes('--release')
const passthroughArgs = process.argv.slice(2).filter((arg) => arg !== '--release')
const args = ['build', '--platform', '--features', features, ...passthroughArgs]

if (isRelease) {
  args.push('--release')
}

const result = spawnSync('napi', args, {
  stdio: 'inherit',
  shell: process.platform === 'win32'
})

if (result.error) {
  console.error(result.error.message)
  process.exit(1)
}

process.exit(result.status ?? 1)
