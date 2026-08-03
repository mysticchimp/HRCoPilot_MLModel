# type: ignore
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from .enums import ImportanceLevel


ProficiencyLevel = Literal["beginner", "intermediate", "advanced", "expert"]
LanguageLevel = Literal["basic", "conversational", "professional", "native", "fluent"]
SeniorityLevel = Literal["entry", "mid", "senior", "executive", "c_level"]
RemoteOption = Literal["remote", "hybrid", "on_site", "flexible"]
EmploymentType = Literal["full_time", "part_time", "contract", "contract_to_hire", "internship"]
UrgencyLevel = Literal["immediate", "within_30_days", "within_60_days", "flexible"]
PayFrequency = Literal["hourly", "annually", "monthly"]
CompanySize = Literal["startup", "small", "medium", "large", "enterprise"]
CompanyStage = Literal["early_stage", "growth", "mature", "public"]
EnvironmentType = Literal["cloud", "on_premise", "hybrid"]

class Skill(BaseModel):
    skill: str = Field(..., description="Name of the skill", example="System Architecture") 
    priority: ImportanceLevel = Field(..., description="Importance level of the skill")
    proficiency_level: Optional[ProficiencyLevel] = Field(None, description="Required proficiency level")


class Technology(BaseModel):
    technology: str = Field(..., description="Technology name", example="AWS")
    priority: ImportanceLevel = Field(..., description="Importance level of the technology")
    version: Optional[str] = Field(None, description="Specific version if applicable", example="Python 3.9+")


class LanguageProficiency(BaseModel):
    language: str = Field(..., description="Language name", example="English") 
    level: LanguageLevel = Field(..., description="Required proficiency level")
    priority: ImportanceLevel = Field(..., description="Importance level of the language")


class Education(BaseModel):
    degree: str = Field(..., description="Degree name", example="Bachelor's in Computer Science") 
    field: Optional[str] = Field(None, description="Field of study", example="Computer Science")
    priority: ImportanceLevel = Field(..., description="Importance level of the degree")


class Certification(BaseModel):
    certification: str = Field(..., description="Certification name", example="CISSP")
    priority: ImportanceLevel = Field(..., description="Importance level of the certification")
    timeline: Optional[str] = Field(None, description="Timeframe to obtain certification", example="within 6 months")


class Qualifications(BaseModel):
    education: Optional[List[Education]] = None
    certifications: Optional[List[Certification]] = None


class ExperienceRange(BaseModel):
    min: Optional[int] = Field(None, description="Minimum years")
    max: Optional[int] = Field(None, description="Maximum years")


class IndustryExperience(BaseModel):
    industry: str = Field(..., description="Industry name", example="fintech")
    priority: ImportanceLevel = Field(..., description="Importance level")


class Leadership(BaseModel):
    required: Optional[bool] = Field(None, description="Whether leadership experience is required")
    team_size: Optional[ExperienceRange] = None
    priority: Optional[ImportanceLevel] = Field(None, description="Importance level of leadership experience")


class Experience(BaseModel):
    level: Optional[SeniorityLevel] = None
    years_total: Optional[ExperienceRange] = None
    years_relevant: Optional[ExperienceRange] = None
    industry_experience: Optional[List[IndustryExperience]] = None
    leadership: Optional[Leadership] = None


class LocationTravelRequirements(BaseModel):
    percentage: Optional[int] = Field(None, description="Travel percentage required", example=25)
    frequency: Optional[str] = Field(None, description="Travel frequency", example="quarterly client visits")


class Relocation(BaseModel):
    assistance_available: Optional[bool] = Field(None, description="Whether relocation assistance is available")
    required: Optional[bool] = Field(None, description="Whether relocation is mandatory")


class Location(BaseModel):
    cities: Optional[List[str]] = Field(None, description="Acceptable cities")
    countries: Optional[List[str]] = Field(None, description="Acceptable countries")
    remote_options: Optional[RemoteOption] = Field(None, description="Work arrangement options")
    travel_requirements: Optional[LocationTravelRequirements] = None
    relocation: Optional[Relocation] = None


class SalaryRange(BaseModel):
    min: Optional[float] = Field(None, description="Minimum salary")
    max: Optional[float] = Field(None, description="Maximum salary")
    currency: Optional[str] = Field(None, description="Currency code", example="USD")
    frequency: Optional[PayFrequency] = Field(None, description="Pay frequency")


class EmploymentDetails(BaseModel):
    type: Optional[EmploymentType] = None
    urgency: Optional[UrgencyLevel] = None
    salary_range: Optional[SalaryRange] = None
    benefits: Optional[List[str]] = Field(None, description="Benefits and perks")


class SoftSkill(BaseModel):
    skill: str = Field(..., description="Soft skill name", example="Communication")
    priority: ImportanceLevel = Field(..., description="Importance level")
    context: Optional[str] = Field(None, description="Context or application", example="client-facing communication")


class ProjectContext(BaseModel):
    types: Optional[List[str]] = Field(None, description="Types of projects", example=["migration", "compliance"])
    methodologies: Optional[List[str]] = Field(None, description="Work methodologies", example=["agile", "devops"])
    environment: Optional[EnvironmentType] = Field(None, description="Technical environment")


class TeamContext(BaseModel):
    size: Optional[int] = Field(None, description="Team size")
    structure: Optional[str] = Field(None, description="Team structure", example="cross-functional agile team")
    reporting_to: Optional[str] = Field(None, description="Reporting role", example="Director of Security")


class SecurityClearance(BaseModel):
    required: Optional[bool] = Field(None, description="Whether clearance is required")
    level: Optional[str] = Field(None, description="Clearance level", example="Secret")


class ComplianceLegal(BaseModel):
    visa_sponsorship: Optional[bool] = Field(None, description="Whether visa sponsorship is available")
    security_clearance: Optional[SecurityClearance] = None
    background_check: Optional[bool] = Field(None, description="Whether background check is required")


class CulturalFit(BaseModel):
    company_values: Optional[List[str]] = Field(None, description="Company values", example=["innovation"])
    work_style: Optional[List[str]] = Field(None, description="Preferred work styles", example=["detail-oriented"])


class Company(BaseModel):
    name: str = Field(..., description="Company name", example="TechCorp Inc.")
    size: Optional[CompanySize] = Field(None, description="Company size category")
    stage: Optional[CompanyStage] = Field(None, description="Company maturity stage")


class JobRoleSchema(BaseModel):
    role: str = Field(..., description="Job title or role name", example="Senior Security Administrator")
    company: Company
    industry: Optional[List[str]] = Field(None, description="Industry sectors", example=["technology", "healthcare"])
    role_objectives: Optional[List[str]] = Field(None, description="High-level role objectives")
    responsibilities: List[str] = Field(..., description="Specific job responsibilities")
    skills: List[Skill] = Field(..., description="Technical and soft skills required")
    technologies: Optional[List[Technology]] = Field(None, description="Required technologies or tools")
    language_proficiency: Optional[List[LanguageProficiency]] = None
    qualifications: Optional[Qualifications] = None
    experience: Optional[Experience] = None
    location: Optional[Location] = None
    employment_details: Optional[EmploymentDetails] = None
    soft_skills: Optional[List[SoftSkill]] = None
    project_context: Optional[ProjectContext] = None
    team_context: Optional[TeamContext] = None
    compliance_legal: Optional[ComplianceLegal] = None
    cultural_fit: Optional[CulturalFit] = None
    growth_opportunities: Optional[List[str]] = Field(None, description="Mentioned career growth opportunities")