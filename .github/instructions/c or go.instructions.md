---
applyTo: '**'
---
the code should be clean and should prefer SOA (Structure Of Arrays) over AOS (Array Of Structures) where applicable. Code should be structured to be eassily vectorizable. Do not allocate memory in not needed reuse it. Also if possible use stack memory over heap memory where applicable use Arenas or custom allocators where applicable. Avoid using exceptions for control flow. Use error codes or optional types instead.