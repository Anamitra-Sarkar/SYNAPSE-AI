CREATE TABLE `projects` (
  `id` int AUTO_INCREMENT NOT NULL,
  `userId` int NOT NULL,
  `title` varchar(160) NOT NULL,
  `status` enum('draft','exploring','blueprint','archived') NOT NULL DEFAULT 'draft',
  `activeBlueprintId` int,
  `createdAt` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updatedAt` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT `projects_id` PRIMARY KEY (`id`),
  INDEX `projects_user_created_idx` (`userId`,`createdAt`)
);

CREATE TABLE `briefs` (
  `id` int AUTO_INCREMENT NOT NULL,
  `projectId` int NOT NULL,
  `userId` int NOT NULL,
  `content` json NOT NULL,
  `createdAt` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updatedAt` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT `briefs_id` PRIMARY KEY (`id`),
  INDEX `briefs_project_idx` (`projectId`),
  INDEX `briefs_user_idx` (`userId`)
);

CREATE TABLE `generationRuns` (
  `id` int AUTO_INCREMENT NOT NULL,
  `projectId` int NOT NULL,
  `briefId` int NOT NULL,
  `userId` int NOT NULL,
  `status` enum('pending','complete','failed') NOT NULL DEFAULT 'pending',
  `recipe` json NOT NULL,
  `rawModelOutput` json,
  `errorSummary` text,
  `createdAt` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `completedAt` timestamp,
  CONSTRAINT `generationRuns_id` PRIMARY KEY (`id`),
  INDEX `runs_project_idx` (`projectId`),
  INDEX `runs_user_idx` (`userId`)
);

CREATE TABLE `concepts` (
  `id` int AUTO_INCREMENT NOT NULL,
  `projectId` int NOT NULL,
  `generationRunId` int NOT NULL,
  `userId` int NOT NULL,
  `rank` int NOT NULL,
  `content` json NOT NULL,
  `rawModelOutput` json,
  `createdAt` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT `concepts_id` PRIMARY KEY (`id`),
  INDEX `concepts_project_rank_idx` (`projectId`,`rank`),
  INDEX `concepts_run_idx` (`generationRunId`)
);

CREATE TABLE `blueprints` (
  `id` int AUTO_INCREMENT NOT NULL,
  `projectId` int NOT NULL,
  `conceptId` int NOT NULL,
  `generationRunId` int NOT NULL,
  `userId` int NOT NULL,
  `rawModelOutput` json NOT NULL,
  `createdAt` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT `blueprints_id` PRIMARY KEY (`id`),
  INDEX `blueprints_project_idx` (`projectId`),
  INDEX `blueprints_concept_idx` (`conceptId`)
);

CREATE TABLE `blueprintEdits` (
  `id` int AUTO_INCREMENT NOT NULL,
  `blueprintId` int NOT NULL,
  `userId` int NOT NULL,
  `content` json NOT NULL,
  `updatedAt` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT `blueprintEdits_id` PRIMARY KEY (`id`),
  INDEX `blueprint_edits_blueprint_idx` (`blueprintId`),
  INDEX `blueprint_edits_user_idx` (`userId`)
);

CREATE TABLE `savedComparisons` (
  `id` int AUTO_INCREMENT NOT NULL,
  `projectId` int NOT NULL,
  `userId` int NOT NULL,
  `conceptIds` json NOT NULL,
  `createdAt` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT `savedComparisons_id` PRIMARY KEY (`id`),
  INDEX `comparisons_project_idx` (`projectId`),
  INDEX `comparisons_user_idx` (`userId`)
);

CREATE TABLE `exports` (
  `id` int AUTO_INCREMENT NOT NULL,
  `projectId` int NOT NULL,
  `blueprintId` int NOT NULL,
  `userId` int NOT NULL,
  `format` varchar(24) NOT NULL DEFAULT 'markdown',
  `createdAt` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT `exports_id` PRIMARY KEY (`id`),
  INDEX `exports_project_idx` (`projectId`),
  INDEX `exports_user_idx` (`userId`)
);
