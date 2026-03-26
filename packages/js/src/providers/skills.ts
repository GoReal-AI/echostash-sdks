import type { SkillDefinition, ToolDefinition } from '../types.js';

export function buildSkillLoadingTool(skills: SkillDefinition[]): ToolDefinition {
  return {
    type: 'function',
    function: {
      name: 'echostash_load_skill',
      description: 'Load a skill to get specialized instructions. Available skills: ' + skills.map(s => s.skill.name).join(', '),
      parameters: {
        type: 'object',
        properties: {
          skill_name: {
            type: 'string',
            enum: skills.map(s => s.skill.name),
            description: 'Name of the skill to load',
          },
        },
        required: ['skill_name'],
      },
    },
  };
}

export function mergeToolsWithSkills(
  tools: ToolDefinition[] | undefined,
  skills: SkillDefinition[] | undefined,
): ToolDefinition[] | undefined {
  if (!skills || skills.length === 0) {
    return tools && tools.length > 0 ? tools : undefined;
  }
  return [...(tools ?? []), buildSkillLoadingTool(skills)];
}
