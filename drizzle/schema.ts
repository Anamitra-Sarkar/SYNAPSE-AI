import { index, int, json, mysqlEnum, mysqlTable, text, timestamp, varchar } from "drizzle-orm/mysql-core";

/**
 * Core user table backing auth flow.
 * Extend this file with additional tables as your product grows.
 * Columns use camelCase to match both database fields and generated types.
 */
export const users = mysqlTable("users", {
  /**
   * Surrogate primary key. Auto-incremented numeric value managed by the database.
   * Use this for relations between tables.
   */
  id: int("id").autoincrement().primaryKey(),
  /** Manus OAuth identifier (openId) returned from the OAuth callback. Unique per user. */
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

export const projects = mysqlTable("projects", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  title: varchar("title", { length: 160 }).notNull(),
  status: mysqlEnum("status", ["draft", "exploring", "blueprint", "archived"]).default("draft").notNull(),
  activeBlueprintId: int("activeBlueprintId"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
}, table => [index("projects_user_created_idx").on(table.userId, table.createdAt)]);

export const briefs = mysqlTable("briefs", {
  id: int("id").autoincrement().primaryKey(),
  projectId: int("projectId").notNull(),
  userId: int("userId").notNull(),
  content: json("content").$type<Record<string, unknown>>().notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
}, table => [index("briefs_project_idx").on(table.projectId), index("briefs_user_idx").on(table.userId)]);

export const generationRuns = mysqlTable("generationRuns", {
  id: int("id").autoincrement().primaryKey(),
  projectId: int("projectId").notNull(),
  briefId: int("briefId").notNull(),
  userId: int("userId").notNull(),
  status: mysqlEnum("status", ["pending", "complete", "failed"]).default("pending").notNull(),
  recipe: json("recipe").$type<Record<string, unknown>>().notNull(),
  rawModelOutput: json("rawModelOutput").$type<unknown>(),
  errorSummary: text("errorSummary"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  completedAt: timestamp("completedAt"),
}, table => [index("runs_project_idx").on(table.projectId), index("runs_user_idx").on(table.userId)]);

export const concepts = mysqlTable("concepts", {
  id: int("id").autoincrement().primaryKey(),
  projectId: int("projectId").notNull(),
  generationRunId: int("generationRunId").notNull(),
  userId: int("userId").notNull(),
  rank: int("rank").notNull(),
  content: json("content").$type<Record<string, unknown>>().notNull(),
  rawModelOutput: json("rawModelOutput").$type<unknown>(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
}, table => [index("concepts_project_rank_idx").on(table.projectId, table.rank), index("concepts_run_idx").on(table.generationRunId)]);

export const blueprints = mysqlTable("blueprints", {
  id: int("id").autoincrement().primaryKey(),
  projectId: int("projectId").notNull(),
  conceptId: int("conceptId").notNull(),
  generationRunId: int("generationRunId").notNull(),
  userId: int("userId").notNull(),
  rawModelOutput: json("rawModelOutput").$type<Record<string, unknown>>().notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
}, table => [index("blueprints_project_idx").on(table.projectId), index("blueprints_concept_idx").on(table.conceptId)]);

export const blueprintEdits = mysqlTable("blueprintEdits", {
  id: int("id").autoincrement().primaryKey(),
  blueprintId: int("blueprintId").notNull(),
  userId: int("userId").notNull(),
  content: json("content").$type<Record<string, unknown>>().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
}, table => [index("blueprint_edits_blueprint_idx").on(table.blueprintId), index("blueprint_edits_user_idx").on(table.userId)]);

export const savedComparisons = mysqlTable("savedComparisons", {
  id: int("id").autoincrement().primaryKey(),
  projectId: int("projectId").notNull(),
  userId: int("userId").notNull(),
  conceptIds: json("conceptIds").$type<number[]>().notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
}, table => [index("comparisons_project_idx").on(table.projectId), index("comparisons_user_idx").on(table.userId)]);

export const exports = mysqlTable("exports", {
  id: int("id").autoincrement().primaryKey(),
  projectId: int("projectId").notNull(),
  blueprintId: int("blueprintId").notNull(),
  userId: int("userId").notNull(),
  format: varchar("format", { length: 24 }).default("markdown").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
}, table => [index("exports_project_idx").on(table.projectId), index("exports_user_idx").on(table.userId)]);
