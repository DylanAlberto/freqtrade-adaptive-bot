"""Fix remaining _partial_exits reference."""
with open('user_data/strategies/MultiPairAdaptiveStrategy.py', 'r') as f:
    content = f.read()

content = content.replace(
    'self._partial_exits.add(partial_key)\n            logger.info(',
    'logger.info('
)

with open('user_data/strategies/MultiPairAdaptiveStrategy.py', 'w') as f:
    f.write(content)
print('Fixed')
